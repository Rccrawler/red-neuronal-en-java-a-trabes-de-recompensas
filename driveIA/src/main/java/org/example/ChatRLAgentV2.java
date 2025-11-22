package org.example;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.json.JSONArray;
import org.json.JSONObject;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class ChatRLAgentV2 {

    // Datos cargados desde JSON
    private static final Map<String, String> RESPUESTAS_CORRECTAS = new HashMap<>();
    private static final List<String> RESPUESTAS_UNICAS = new ArrayList<>();

    // Parámetros de la red
    private static final int STATE_SIZE = 26;  // 26 letras
    private static final double GAMMA = 0.99;
    private static final double EPSILON_DECAY = 0.995;
    private static final double LEARNING_RATE = 0.001;

    private double epsilon = 1.0;
    private Random random = new Random();
    private MultiLayerNetwork model;
    private int totalAciertos = 0;
    private int totalIntentosEntrenamiento = 0;
    private Map<String, Integer> respuestaAIndice = new HashMap<>();
    private int ACTIONS;

    public ChatRLAgentV2() {
        // Se inicializa después de cargar JSON
    }

    /**
     * Carga las preguntas y respuestas desde un archivo JSON
     */
    public static void cargarDesdeJSON(String rutaArchivo) {
        try {
            String contenido = new String(Files.readAllBytes(Paths.get(rutaArchivo)));
            JSONArray jsonArray = new JSONArray(contenido);

            RESPUESTAS_CORRECTAS.clear();
            RESPUESTAS_UNICAS.clear();

            for (int i = 0; i < jsonArray.length(); i++) {
                JSONObject obj = jsonArray.getJSONObject(i);
                String pregunta = obj.getString("pregunta").toLowerCase().trim();
                String respuesta = obj.getString("respuesta").toLowerCase().trim();
                RESPUESTAS_CORRECTAS.put(pregunta, respuesta);

                // Agregar respuesta única si no existe
                if (!RESPUESTAS_UNICAS.contains(respuesta)) {
                    RESPUESTAS_UNICAS.add(respuesta);
                }
            }

            System.out.println("✅ Se cargaron " + RESPUESTAS_CORRECTAS.size() + " pares pregunta-respuesta desde JSON");
            System.out.println("📊 Respuestas únicas: " + RESPUESTAS_UNICAS.size() + "\n");

        } catch (IOException e) {
            System.err.println("❌ Error al leer el archivo JSON: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Aprende una nueva pregunta-respuesta en tiempo real
     */
    public void aprenderNuevoParPreguntaRespuesta(String pregunta, String respuesta) {
        pregunta = pregunta.toLowerCase().trim();
        respuesta = respuesta.toLowerCase().trim();

        // Agregar al dataset si no existe
        boolean preguntaNueva = !RESPUESTAS_CORRECTAS.containsKey(pregunta);
        RESPUESTAS_CORRECTAS.put(pregunta, respuesta);

        // Agregar respuesta única si no existe
        boolean respuestaNueva = !RESPUESTAS_UNICAS.contains(respuesta);
        if (respuestaNueva) {
            RESPUESTAS_UNICAS.add(respuesta);
            respuestaAIndice.put(respuesta, RESPUESTAS_UNICAS.size() - 1);
            ACTIONS = RESPUESTAS_UNICAS.size();

            // Reconstruir red con más neuronas de salida
            System.out.println("🔄 Nueva respuesta detectada. Expandiendo red a " + ACTIONS + " salidas...");
            inicializarRed();

            // RE-ENTRENAR TODO EL DATASET para recuperar el conocimiento
            System.out.println("🔄 Re-entrenando todo el dataset para mantener conocimiento...");
            double epsilonOriginal = this.epsilon;
            this.epsilon = 0.1; // Poca exploración para re-entrenamiento

            for (int i = 0; i < 20; i++) { // 20 episodios de re-entrenamiento
                for (Map.Entry<String, String> entry : RESPUESTAS_CORRECTAS.entrySet()) {
                    String p = entry.getKey();
                    String r = entry.getValue();
                    String rGen = selectResponse(p);
                    learn(p, rGen, r);
                }
            }
            this.epsilon = epsilonOriginal;
        } else {
            // Si la respuesta ya existe, solo entrenar esta pregunta varias veces
            for (int i = 0; i < 50; i++) {
                String respuestaGenerada = selectResponse(pregunta);
                learn(pregunta, respuestaGenerada, respuesta);
            }
        }

        // Entrenar intensivamente el nuevo par pregunta-respuesta
        double recompensaFinal = 0;
        for (int i = 0; i < 100; i++) {
            String respuestaGenerada = selectResponse(pregunta);
            recompensaFinal = calcularRecompensa(pregunta, respuestaGenerada, respuesta);
            learn(pregunta, respuestaGenerada, respuesta);
        }

        System.out.println("📝 Aprendido: \"" + pregunta + "\" → \"" + respuesta + "\" | Recompensa final: " + String.format("%.1f", recompensaFinal) + "/10");
    }

    /**
     * Inicializa la red neuronal dinámicamente según el número de respuestas únicas
     */
    public void inicializarRed() {
        this.ACTIONS = RESPUESTAS_UNICAS.size();

        // Crear mapeos
        for (int i = 0; i < RESPUESTAS_UNICAS.size(); i++) {
            respuestaAIndice.put(RESPUESTAS_UNICAS.get(i), i);
        }

        // Construir red neuronal dinámicamente
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .updater(new Adam(LEARNING_RATE))
                .list()
                .layer(new DenseLayer.Builder().nIn(STATE_SIZE).nOut(256)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nOut(128)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nOut(64)
                        .activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(ACTIONS).build())
                .build();

        model = new MultiLayerNetwork(config);
        model.init();
    }

    /**
     * Convierte una pregunta a vector: UNA NEURONA POR LETRA
     */
    private double[] preguntaAVector(String pregunta) {
        double[] vector = new double[STATE_SIZE];
        pregunta = pregunta.toLowerCase().replaceAll("[^a-z]", "");

        for (char c : pregunta.toCharArray()) {
            if (c >= 'a' && c <= 'z') {
                int indice = c - 'a';
                vector[indice] = 1.0;
            }
        }

        return vector;
    }

    /**
     * Selecciona una respuesta basada en la pregunta
     */
    public String selectResponse(String pregunta) {
        if (random.nextDouble() < epsilon) {
            return RESPUESTAS_UNICAS.get(random.nextInt(RESPUESTAS_UNICAS.size()));
        }

        double[] preguntaVector = preguntaAVector(pregunta);
        INDArray input = Nd4j.create(new double[][]{preguntaVector});
        INDArray output = model.output(input);
        int respuestaIndex = Nd4j.argMax(output, 1).getInt(0);

        return RESPUESTAS_UNICAS.get(respuestaIndex);
    }

    /**
     * Calcula la recompensa comparando respuesta generada vs correcta
     */
    private double calcularRecompensa(String pregunta, String respuestaGenerada, String respuestaCorrecta) {
        respuestaGenerada = respuestaGenerada.toLowerCase().trim();
        respuestaCorrecta = respuestaCorrecta.toLowerCase().trim();

        if (respuestaGenerada.equals(respuestaCorrecta)) {
            return 10.0;
        }

        String[] palabrasGeneradas = respuestaGenerada.split("\\s+");
        String[] palabrasCorrectas = respuestaCorrecta.split("\\s+");

        int coincidencias = 0;
        for (String pGen : palabrasGeneradas) {
            for (String pCorr : palabrasCorrectas) {
                if (pGen.equals(pCorr)) {
                    coincidencias++;
                    break;
                }
            }
        }

        double recompensa = ((double) coincidencias / palabrasCorrectas.length) * 10;
        return Math.max(-1, recompensa);
    }

    /**
     * El agente aprende de la pregunta y la respuesta
     */
    public void learn(String pregunta, String respuestaGenerada, String respuestaCorrecta) {
        double[] preguntaVector = preguntaAVector(pregunta);
        INDArray input = Nd4j.create(new double[][]{preguntaVector});

        double[] targetArray = new double[ACTIONS];
        Integer indiceCorrecta = respuestaAIndice.get(respuestaCorrecta);
        if (indiceCorrecta == null) {
            indiceCorrecta = 0;
        }
        targetArray[indiceCorrecta] = 1.0;

        INDArray target = Nd4j.create(new double[][]{targetArray});

        double reward = calcularRecompensa(pregunta, respuestaGenerada, respuestaCorrecta);

        model.fit(input, target);
        epsilon *= EPSILON_DECAY;

        if (reward > 5) {
            totalAciertos++;
        }
        totalIntentosEntrenamiento++;
    }

    public void disableExploration() {
        epsilon = 0.0;
    }

    public static void main(String[] args) {
        // Cargar preguntas desde JSON
        String rutaJSON = "preguntas.json";
        cargarDesdeJSON(rutaJSON);

        ChatRLAgentV2 agent = new ChatRLAgentV2();
        agent.inicializarRed();

        System.out.println("╔═══════════════════════════════════════╗");
        System.out.println("║   CHATBOT CON RL - DINÁMICO           ║");
        System.out.println("║   (Adapta respuestas según JSON)      ║");
        System.out.println("╚═══════════════════════════════════════╝\n");

        System.out.println("🧠 Arquitectura de la red:");
        System.out.println("  Entrada: 26 neuronas (una por letra a-z)");
        System.out.println("  Capa 1:  256 neuronas (RELU)");
        System.out.println("  Capa 2:  128 neuronas (RELU)");
        System.out.println("  Capa 3:  64 neuronas (RELU)");
        System.out.println("  Salida:  " + agent.ACTIONS + " neuronas (SOFTMAX + Cross-Entropy)\n");

        System.out.println("⚙️ Parámetros:");
        System.out.println("  Learning Rate: " + ChatRLAgentV2.LEARNING_RATE);
        System.out.println("  Epsilon Decay: " + ChatRLAgentV2.EPSILON_DECAY);
        System.out.println("  Gamma: " + ChatRLAgentV2.GAMMA + "\n");

        // ========== ENTRENAMIENTO ==========
        System.out.println("📚 INICIANDO ENTRENAMIENTO...\n");

        int episodios = 100;

        for (int ep = 1; ep <= episodios; ep++) {
            double recompensaTotalEpisodio = 0;
            int intentosEpisodio = 0;

            for (Map.Entry<String, String> entry : RESPUESTAS_CORRECTAS.entrySet()) {
                String pregunta = entry.getKey();
                String respuestaCorrecta = entry.getValue();

                String respuestaGenerada = agent.selectResponse(pregunta);
                double recompensa = agent.calcularRecompensa(pregunta, respuestaGenerada, respuestaCorrecta);
                agent.learn(pregunta, respuestaGenerada, respuestaCorrecta);

                recompensaTotalEpisodio += recompensa;
                intentosEpisodio++;
            }

            if (ep % 10 == 0) {
                double porcentajeAcierto = (agent.totalAciertos * 100.0) / agent.totalIntentosEntrenamiento;
                double recompensaPromedio = recompensaTotalEpisodio / intentosEpisodio;
                System.out.println("Episodio " + ep + "/" + episodios +
                                 " | Precisión: " + String.format("%.1f", porcentajeAcierto) + "% | " +
                                 "Recompensa: " + String.format("%.2f", recompensaPromedio) + "/10 | " +
                                 "ε: " + String.format("%.4f", agent.epsilon));
            }
        }

        System.out.println("\n✅ Entrenamiento completado.\n");

        // ========== PRUEBA ==========
        System.out.println("🧪 PRUEBA DEL CHATBOT ENTRENADO\n");

        agent.disableExploration();

        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║      🤖 CHATBOT RL EN ACCIÓN 🤖        ║");
        System.out.println("╚════════════════════════════════════════╝\n");

        int aciertosFinales = 0;
        // MOSTRAR TODAS LAS PREGUNTAS, NO SOLO 20
        List<String> todasLasPreguntas = new ArrayList<>(RESPUESTAS_CORRECTAS.keySet());

        for (String pregunta : todasLasPreguntas) {
            String respuestaCorrecta = RESPUESTAS_CORRECTAS.get(pregunta);
            String respuestaGenerada = agent.selectResponse(pregunta);

            double reward = agent.calcularRecompensa(pregunta, respuestaGenerada, respuestaCorrecta);
            String estado = (reward > 5) ? "✅ CORRECTO" : "❌ INCORRECTO";
            if (reward > 5) aciertosFinales++;

            System.out.println("Pregunta:          \"" + pregunta + "\"");
            System.out.println("Esperada:          \"" + respuestaCorrecta + "\"");
            System.out.println("Generada:          \"" + respuestaGenerada + "\"");
            System.out.println("Puntuación:        " + String.format("%.1f", reward) + "/10  " + estado);
            System.out.println("─────────────────────────────────────────\n");
        }

        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║          📊 RESULTADOS FINALES         ║");
        System.out.println("║  Aciertos: " + aciertosFinales + "/" + todasLasPreguntas.size() + "  (" + String.format("%.1f", (aciertosFinales * 100.0) / todasLasPreguntas.size()) + "%)");
        System.out.println("╚════════════════════════════════════════╝\n\n");

        // ========== APRENDIZAJE CONTINUO ==========
        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║  🔄 APRENDIZAJE CONTINUO               ║");
        System.out.println("║  Enseñando nuevas preguntas...         ║");
        System.out.println("╚════════════════════════════════════════╝\n");

        // Enseñar nuevos pares pregunta-respuesta
        agent.aprenderNuevoParPreguntaRespuesta("qué tal", "bien gracias");
        agent.aprenderNuevoParPreguntaRespuesta("cómo va todo", "todo bien");
        agent.aprenderNuevoParPreguntaRespuesta("hasta pronto", "nos vemos");
        agent.aprenderNuevoParPreguntaRespuesta("me puedes ayudar", "claro, dime");
        agent.aprenderNuevoParPreguntaRespuesta("qué día es hoy", "no lo sé");

        System.out.println("\n✅ Total respuestas en el sistema: " + RESPUESTAS_UNICAS.size());
        System.out.println("✅ Total pares pregunta-respuesta: " + RESPUESTAS_CORRECTAS.size() + "\n");

        // Probar las nuevas preguntas aprendidas
        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║  🧪 PRUEBA DE PREGUNTAS NUEVAS         ║");
        System.out.println("╚════════════════════════════════════════╝\n");

        List<String> preguntasNuevas = Arrays.asList(
            "qué tal",
            "cómo va todo",
            "hasta pronto",
            "me puedes ayudar",
            "qué día es hoy"
        );

        int aciertosNuevos = 0;
        for (String pregunta : preguntasNuevas) {
            String respuestaEsperada = RESPUESTAS_CORRECTAS.get(pregunta);
            String respuesta = agent.selectResponse(pregunta);
            double reward = agent.calcularRecompensa(pregunta, respuesta, respuestaEsperada);
            String estado = (reward > 5) ? "✅ CORRECTO" : "❌ INCORRECTO";
            if (reward > 5) aciertosNuevos++;

            System.out.println("Pregunta nueva:    \"" + pregunta + "\"");
            System.out.println("Esperada:          \"" + respuestaEsperada + "\"");
            System.out.println("Generada:          \"" + respuesta + "\"");
            System.out.println("Puntuación:        " + String.format("%.1f", reward) + "/10  " + estado);
            System.out.println("─────────────────────────────────────────\n");
        }

        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║  📈 RESULTADOS APRENDIZAJE CONTINUO    ║");
        System.out.println("║  Aciertos nuevas: " + aciertosNuevos + "/" + preguntasNuevas.size() + "  (" + String.format("%.1f", (aciertosNuevos * 100.0) / preguntasNuevas.size()) + "%)");
        System.out.println("╚════════════════════════════════════════╝");
    }
}
