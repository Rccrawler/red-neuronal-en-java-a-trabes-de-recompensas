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

public class ChatRLAgentMejorado {

    // DICCIONARIO de Preguntas -> Respuestas correctas
    private static final Map<String, String> RESPUESTAS_CORRECTAS = new HashMap<>();
    static {
        RESPUESTAS_CORRECTAS.put("hola", "hola como estás");
        RESPUESTAS_CORRECTAS.put("qué tal", "bien gracias");
        RESPUESTAS_CORRECTAS.put("cómo estás", "estoy muy bien");
        RESPUESTAS_CORRECTAS.put("quién eres", "soy un chatbot");
        RESPUESTAS_CORRECTAS.put("ayuda", "claro te ayudaré");
        RESPUESTAS_CORRECTAS.put("buenos días", "buenos días que tal");
        RESPUESTAS_CORRECTAS.put("buenas noches", "buenas noches descansa");
        RESPUESTAS_CORRECTAS.put("gracias", "de nada para eso estoy");
        RESPUESTAS_CORRECTAS.put("adiós", "adiós hasta luego");
        RESPUESTAS_CORRECTAS.put("nombre", "me llamo chatbot");
    }

    // FRECUENCIA DE LETRAS: 26 entradas en lugar de solo presencia/ausencia
    private static final int ACTIONS = 15;  // 15 respuestas posibles
    private static final int STATE_SIZE = 26; // Frecuencia de cada letra (0.0 a 1.0)
    private static final double GAMMA = 0.99;
    private static final double EPSILON_DECAY = 0.995;
    private static final double LEARNING_RATE = 0.001; // Aumentado para mejor convergencia

    private double epsilon = 1.0;
    private Random random = new Random();
    private MultiLayerNetwork model;
    private int totalAciertos = 0;
    private int totalIntentosEntrenamiento = 0;

    // Base de respuestas entrenadas
    private String[] respuestasBase = {
        "hola como estás",
        "bien gracias",
        "estoy muy bien",
        "soy un chatbot",
        "claro te ayudaré",
        "no entiendo",
        "puedo ayudarte",
        "qué necesitas",
        "gracias por preguntar",
        "estoy aquí para ayudarte",
        "buenos días que tal",
        "buenas noches descansa",
        "de nada para eso estoy",
        "adiós hasta luego",
        "me llamo chatbot"
    };

    public ChatRLAgentMejorado() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .updater(new Adam(LEARNING_RATE))
                .list()
                .layer(new DenseLayer.Builder().nIn(STATE_SIZE).nOut(128)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nOut(128)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nOut(64)
                        .activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nOut(ACTIONS).build())
                .build();

        model = new MultiLayerNetwork(config);
        model.init();
    }

    /**
     * Codificación mejorada: FRECUENCIA DE LETRAS
     * "hola" de longitud 4:
     * - 'h' aparece 1 vez → 1/4 = 0.25
     * - 'o' aparece 1 vez → 1/4 = 0.25
     * - 'l' aparece 1 vez → 1/4 = 0.25
     * - 'a' aparece 1 vez → 1/4 = 0.25
     */
    private double[] preguntaAVector(String pregunta) {
        double[] vector = new double[STATE_SIZE];
        pregunta = pregunta.toLowerCase().replaceAll("[^a-z]", "");

        if (pregunta.length() == 0) return vector;

        // Contar frecuencia de cada letra
        int[] frecuencias = new int[26];
        for (char c : pregunta.toCharArray()) {
            if (c >= 'a' && c <= 'z') {
                frecuencias[c - 'a']++;
            }
        }

        // Normalizar por longitud total
        for (int i = 0; i < 26; i++) {
            vector[i] = (double) frecuencias[i] / pregunta.length();
        }

        return vector;
    }

    /**
     * Selecciona respuesta
     */
    public String selectResponse(String pregunta) {
        if (random.nextDouble() < epsilon) {
            return respuestasBase[random.nextInt(respuestasBase.length)];
        }

        double[] preguntaVector = preguntaAVector(pregunta);
        INDArray input = Nd4j.create(new double[][]{preguntaVector});
        INDArray output = model.output(input);
        int respuestaIndex = Nd4j.argMax(output, 1).getInt(0) % respuestasBase.length;

        return respuestasBase[respuestaIndex];
    }

    /**
     * Calcula recompensa
     */
    private double calcularRecompensa(String pregunta, String respuestaGenerada, String respuestaCorrecta) {
        respuestaGenerada = respuestaGenerada.toLowerCase();
        respuestaCorrecta = respuestaCorrecta.toLowerCase();

        String[] palabrasGeneradas = respuestaGenerada.split(" ");
        String[] palabrasCorrectas = respuestaCorrecta.split(" ");

        int coincidencias = 0;
        for (String pGen : palabrasGeneradas) {
            for (String pCorr : palabrasCorrectas) {
                if (pGen.equals(pCorr)) {
                    coincidencias++;
                }
            }
        }

        double recompensa = ((double) coincidencias / palabrasCorrectas.length) * 10;
        return Math.max(-5, recompensa);
    }

    /**
     * Aprende
     */
    public void learn(String pregunta, String respuestaGenerada, String respuestaCorrecta) {
        double[] preguntaVector = preguntaAVector(pregunta);
        double[] respuestaVectorCorrecta = preguntaAVector(respuestaCorrecta);

        INDArray input = Nd4j.create(new double[][]{preguntaVector});
        INDArray target = model.output(input).dup();

        INDArray nextInput = Nd4j.create(new double[][]{respuestaVectorCorrecta});
        INDArray futureQ = model.output(nextInput);

        double reward = calcularRecompensa(pregunta, respuestaGenerada, respuestaCorrecta);
        double qUpdated = reward + GAMMA * futureQ.maxNumber().doubleValue();

        int respuestaIndex = Math.abs(respuestaGenerada.hashCode() % ACTIONS);
        target.putScalar(respuestaIndex, qUpdated);

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
        ChatRLAgentMejorado agent = new ChatRLAgentMejorado();

        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║  📚 CHATBOT RL MEJORADO               ║");
        System.out.println("║  FRECUENCIA DE LETRAS (26 NEURONAS)   ║");
        System.out.println("╚════════════════════════════════════════╝\n");

        System.out.println("🧠 Arquitectura de la red:");
        System.out.println("  Entrada: 26 neuronas (frecuencia de letras)");
        System.out.println("  Capa 1:  128 neuronas (RELU)");
        System.out.println("  Capa 2:  128 neuronas (RELU)");
        System.out.println("  Capa 3:  64 neuronas (RELU)");
        System.out.println("  Salida:  15 neuronas (respuestas)\n");

        System.out.println("📚 INICIANDO ENTRENAMIENTO...\n");

        int episodios = 1000; // Aumentado a 1000

        for (int ep = 1; ep <= episodios; ep++) {
            for (Map.Entry<String, String> entry : RESPUESTAS_CORRECTAS.entrySet()) {
                String pregunta = entry.getKey();
                String respuestaCorrecta = entry.getValue();
                String respuestaGenerada = agent.selectResponse(pregunta);
                agent.learn(pregunta, respuestaGenerada, respuestaCorrecta);
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
        for (Map.Entry<String, String> entry : RESPUESTAS_CORRECTAS.entrySet()) {
            String pregunta = entry.getKey();
            String respuestaCorrecta = entry.getValue();
            String respuestaGenerada = agent.selectResponse(pregunta);

            double reward = agent.calcularRecompensa(pregunta, respuestaGenerada, respuestaCorrecta);
            String estado = (reward > 5) ? "✅ CORRECTO" : "❌ INCORRECTO";
            if (reward > 5) aciertosFinales++;

            System.out.println("Pregunta:    \"" + pregunta + "\"");
            System.out.println("Esperada:    \"" + respuestaCorrecta + "\"");
            System.out.println("Generada:    \"" + respuestaGenerada + "\"");
            System.out.println("Puntuación:  " + String.format("%.1f", reward) + "/10  " + estado);
            System.out.println("─────────────────────────────────────────\n");
        }

        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║          📊 RESULTADOS FINALES         ║");
        System.out.println("║  Aciertos: " + aciertosFinales + "/" + RESPUESTAS_CORRECTAS.size() +
                         "  (" + String.format("%.1f", (aciertosFinales * 100.0) / RESPUESTAS_CORRECTAS.size()) + "%)");
        System.out.println("╚════════════════════════════════════════╝");
    }
}
