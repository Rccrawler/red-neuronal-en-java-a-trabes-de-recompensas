package org.example;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class ChatRLAgentOptimo {

    // DICCIONARIO con índices para mejor mapeo
    private static final Map<String, Integer> MAPEO_PREGUNTAS = new HashMap<>();
    private static final Map<Integer, String> PREGUNTAS_INVERSO = new HashMap<>();
    private static final Map<Integer, String> RESPUESTAS_MAPEADAS = new HashMap<>();

    static {
        int idx = 0;
        MAPEO_PREGUNTAS.put("hola", idx);
        PREGUNTAS_INVERSO.put(idx, "hola");
        RESPUESTAS_MAPEADAS.put(idx, "hola como estás");
        idx++;

        MAPEO_PREGUNTAS.put("qué tal", idx);
        PREGUNTAS_INVERSO.put(idx, "qué tal");
        RESPUESTAS_MAPEADAS.put(idx, "bien gracias");
        idx++;

        MAPEO_PREGUNTAS.put("cómo estás", idx);
        PREGUNTAS_INVERSO.put(idx, "cómo estás");
        RESPUESTAS_MAPEADAS.put(idx, "estoy muy bien");
        idx++;

        MAPEO_PREGUNTAS.put("quién eres", idx);
        PREGUNTAS_INVERSO.put(idx, "quién eres");
        RESPUESTAS_MAPEADAS.put(idx, "soy un chatbot");
        idx++;

        MAPEO_PREGUNTAS.put("ayuda", idx);
        PREGUNTAS_INVERSO.put(idx, "ayuda");
        RESPUESTAS_MAPEADAS.put(idx, "claro te ayudaré");
        idx++;

        MAPEO_PREGUNTAS.put("buenos días", idx);
        PREGUNTAS_INVERSO.put(idx, "buenos días");
        RESPUESTAS_MAPEADAS.put(idx, "buenos días que tal");
        idx++;

        MAPEO_PREGUNTAS.put("buenas noches", idx);
        PREGUNTAS_INVERSO.put(idx, "buenas noches");
        RESPUESTAS_MAPEADAS.put(idx, "buenas noches descansa");
        idx++;

        MAPEO_PREGUNTAS.put("gracias", idx);
        PREGUNTAS_INVERSO.put(idx, "gracias");
        RESPUESTAS_MAPEADAS.put(idx, "de nada para eso estoy");
        idx++;

        MAPEO_PREGUNTAS.put("adiós", idx);
        PREGUNTAS_INVERSO.put(idx, "adiós");
        RESPUESTAS_MAPEADAS.put(idx, "adiós hasta luego");
        idx++;

        MAPEO_PREGUNTAS.put("nombre", idx);
        PREGUNTAS_INVERSO.put(idx, "nombre");
        RESPUESTAS_MAPEADAS.put(idx, "me llamo chatbot");
    }

    private static final int NUM_PREGUNTAS = MAPEO_PREGUNTAS.size();
    private static final int STATE_SIZE = 26; // Frecuencia de letras
    private static final double GAMMA = 0.99;
    private static final double EPSILON_DECAY = 0.995;
    private static final double LEARNING_RATE = 0.0001;

    private double epsilon = 1.0;
    private Random random = new Random();
    private MultiLayerNetwork model;
    private int totalAciertos = 0;
    private int totalIntentosEntrenamiento = 0;

    public ChatRLAgentOptimo() {
        // Red neuronal para clasificar preguntas
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .updater(new Adam(LEARNING_RATE))
                .list()
                .layer(new DenseLayer.Builder().nIn(STATE_SIZE).nOut(192)// original 64
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nOut(96) // original 32
                        .activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nOut(NUM_PREGUNTAS).build())
                .build();

        model = new MultiLayerNetwork(config);
        model.init();
    }

    /**
     * Convierte pregunta a vector de frecuencia de letras
     */
    private double[] preguntaAVector(String pregunta) {
        double[] vector = new double[STATE_SIZE];
        pregunta = pregunta.toLowerCase().replaceAll("[^a-z]", "");

        if (pregunta.length() == 0) return vector;

        int[] frecuencias = new int[26];
        for (char c : pregunta.toCharArray()) {
            if (c >= 'a' && c <= 'z') {
                frecuencias[c - 'a']++;
            }
        }

        for (int i = 0; i < 26; i++) {
            vector[i] = (double) frecuencias[i] / pregunta.length();
        }

        return vector;
    }

    /**
     * Selecciona respuesta basada en clasificación directa
     */
    public String selectResponse(String pregunta) {
        if (random.nextDouble() < epsilon) {
            // Exploración: pregunta aleatoria
            int randomIdx = random.nextInt(NUM_PREGUNTAS);
            return RESPUESTAS_MAPEADAS.get(randomIdx);
        }

        // Explotación: usa la red neuronal
        double[] preguntaVector = preguntaAVector(pregunta);
        INDArray input = Nd4j.create(new double[][]{preguntaVector});
        INDArray output = model.output(input);
        int mejorClase = Nd4j.argMax(output, 1).getInt(0);

        return RESPUESTAS_MAPEADAS.getOrDefault(mejorClase, "no entiendo");
    }

    /**
     * Aprende: compara respuesta generada vs correcta
     */
    public void learn(String pregunta, int indiceCorrectoEsperado) {
        double[] preguntaVector = preguntaAVector(pregunta);

        INDArray input = Nd4j.create(new double[][]{preguntaVector});
        INDArray output = model.output(input);

        // Obtener la clase predicha
        int clasePredicada = Nd4j.argMax(output, 1).getInt(0);

        // Recompensa: +10 si acierta, -1 si falla
        double reward = (clasePredicada == indiceCorrectoEsperado) ? 10 : -1;

        // Crear target: la clase correcta debe tener valor alto
        INDArray target = output.dup();
        target.putScalar(indiceCorrectoEsperado, reward);

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
        ChatRLAgentOptimo agent = new ChatRLAgentOptimo();

        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║  🚀 CHATBOT RL ÓPTIMO                 ║");
        System.out.println("║  CLASIFICACIÓN DIRECTA DE PREGUNTAS   ║");
        System.out.println("╚════════════════════════════════════════╝\n");

        System.out.println("🧠 Arquitectura de la red:");
        System.out.println("  Entrada:   26 neuronas (frecuencia de letras)");
        System.out.println("  Capa 1:    64 neuronas (RELU)");
        System.out.println("  Capa 2:    32 neuronas (RELU)");
        System.out.println("  Salida:    " + NUM_PREGUNTAS + " neuronas (una por pregunta)\n");

        System.out.println("🎯 Estrategia: Clasificación con RL");
        System.out.println("  - La red aprende a CLASIFICAR preguntas");
        System.out.println("  - Cada neurona de salida = una pregunta diferente");
        System.out.println("  - La respuesta se obtiene del mapeo pregunta→respuesta\n");

        System.out.println("📚 INICIANDO ENTRENAMIENTO...\n");

        int episodios = 1000;// original 500

        for (int ep = 1; ep <= episodios; ep++) {
            // Entrenar con cada pregunta-respuesta
            for (Map.Entry<String, Integer> entry : MAPEO_PREGUNTAS.entrySet()) {
                String pregunta = entry.getKey();
                int indiceEsperado = entry.getValue();

                agent.learn(pregunta, indiceEsperado);
            }

            if (ep % 50 == 0) {
                double porcentajeAcierto = (agent.totalAciertos * 100.0) / agent.totalIntentosEntrenamiento;
                System.out.println("Episodio " + ep + "/" + episodios +
                                 " | Precisión: " + String.format("%.1f", porcentajeAcierto) + "% | ε: " + String.format("%.6f", agent.epsilon));
            }
        }

        System.out.println("\n✅ Entrenamiento completado.\n");

        System.out.println("🧪 PRUEBA DEL CHATBOT\n");
        agent.disableExploration();

        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║     🤖 CHATBOT RL EN ACCIÓN 🤖        ║");
        System.out.println("╚════════════════════════════════════════╝\n");

        int aciertosFinales = 0;
        for (Map.Entry<String, Integer> entry : MAPEO_PREGUNTAS.entrySet()) {
            String pregunta = entry.getKey();
            int indiceEsperado = entry.getValue();
            String respuestaEsperada = RESPUESTAS_MAPEADAS.get(indiceEsperado);
            String respuestaGenerada = agent.selectResponse(pregunta);

            boolean esCorrect = respuestaGenerada.equals(respuestaEsperada);
            String estado = esCorrect ? "✅ CORRECTO" : "❌ INCORRECTO";
            if (esCorrect) aciertosFinales++;

            System.out.println("Pregunta:    \"" + pregunta + "\"");
            System.out.println("Esperada:    \"" + respuestaEsperada + "\"");
            System.out.println("Generada:    \"" + respuestaGenerada + "\"");
            System.out.println("Resultado:   " + estado);
            System.out.println("─────────────────────────────────────────\n");
        }

        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║          📊 RESULTADOS FINALES         ║");
        System.out.println("║  Aciertos: " + aciertosFinales + "/" + MAPEO_PREGUNTAS.size() +
                         "  (" + String.format("%.1f", (aciertosFinales * 100.0) / MAPEO_PREGUNTAS.size()) + "%)");
        System.out.println("║                                        ║");
        System.out.println("║  ¡Gracias por tu entusiasmo! 🎉        ║");
        System.out.println("╚════════════════════════════════════════╝");
    }
}
