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

import java.util.Random;

public class TextRLAgent {

    // Para texto: 26 letras (a-z)
    private static final int ACTIONS = 26;  // a, b, c, ..., z
    // Entrada: última letra escrita (1 número: 0-25)
    private static final int STATE_SIZE = 1;
    private static final double GAMMA = 0.99;
    private static final double EPSILON_DECAY = 0.995;
    private static final double LEARNING_RATE = 0.01;

    private double epsilon = 1.0;
    private Random random = new Random();
    private MultiLayerNetwork model;

    // Mapeo de letras a números
    private static final String ALPHABET = "abcdefghijklmnopqrstuvwxyz";

    public TextRLAgent() {
        // Red neuronal más grande para texto
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .updater(new Adam(LEARNING_RATE))
                .list()
                .layer(new DenseLayer.Builder().nIn(STATE_SIZE).nOut(64)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nOut(64)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nOut(32)
                        .activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nOut(ACTIONS).build())
                .build();

        model = new MultiLayerNetwork(config);
        model.init();
    }

    // Convertir letra a número (0-25)
    private int letterToNumber(char letter) {
        return Character.toLowerCase(letter) - 'a';
    }

    // Convertir número a letra (0-25)
    private char numberToLetter(int num) {
        return ALPHABET.charAt(Math.max(0, Math.min(25, num)));
    }

    // Seleccionar siguiente letra
    public int selectAction(int currentLetter) {
        if (random.nextDouble() < epsilon) {
            return random.nextInt(ACTIONS); // Letra aleatoria
        }

        INDArray input = Nd4j.create(new double[][]{{currentLetter}});
        INDArray output = model.output(input);
        return Nd4j.argMax(output, 1).getInt(0);
    }

    // Aprender de la recompensa
    public void learn(int currentLetter, int nextLetter, double reward) {
        INDArray input = Nd4j.create(new double[][]{{currentLetter}});
        INDArray target = model.output(input).dup();

        INDArray nextInput = Nd4j.create(new double[][]{{nextLetter}});
        INDArray futureQ = model.output(nextInput);

        double qUpdated = reward + GAMMA * futureQ.maxNumber().doubleValue();
        target.putScalar(nextLetter, qUpdated);

        model.fit(input, target);
        epsilon *= EPSILON_DECAY;
    }

    // Desactivar exploración para pruebas
    public void disableExploration() {
        epsilon = 0.0;
    }

    // Resetear epsilon
    public void resetEpsilon() {
        epsilon = 1.0;
    }

    public static void main(String[] args) {
        TextRLAgent agent = new TextRLAgent();

        // ENTRENAMIENTO: Enseñar al agente a generar palabras válidas
        System.out.println("📚 ENTRENAMIENTO DEL AGENTE DE TEXTO\n");

        String[] palabras = {"hello", "world", "java", "code", "learn"};
        int episodios = 100;

        for (int ep = 1; ep <= episodios; ep++) {
            for (String palabra : palabras) {
                // Para cada letra en la palabra
                for (int i = 0; i < palabra.length() - 1; i++) {
                    int currentLetter = agent.letterToNumber(palabra.charAt(i));
                    int nextLetter = agent.letterToNumber(palabra.charAt(i + 1));

                    int selectedLetter = agent.selectAction(currentLetter);

                    // Recompensa: +10 si acierta, -1 si falla
                    double reward = (selectedLetter == nextLetter) ? 10 : -1;

                    agent.learn(currentLetter, selectedLetter, reward);
                }
            }

            if (ep % 20 == 0) {
                System.out.println("Episodio " + ep + " completado");
            }
        }

        System.out.println("\n✅ Entrenamiento terminado.\n");

        // PRUEBA: Ver qué palabras genera
        System.out.println("🧪 PRUEBA DEL AGENTE - Generando texto\n");

        agent.disableExploration();

        for (int test = 0; test < 5; test++) {
            StringBuilder generatedWord = new StringBuilder();
            int currentLetter = agent.random.nextInt(26); // Letra inicial aleatoria
            generatedWord.append(agent.numberToLetter(currentLetter));

            // Generar 5 letras más
            for (int i = 0; i < 5; i++) {
                int nextLetter = agent.selectAction(currentLetter);
                generatedWord.append(agent.numberToLetter(nextLetter));
                currentLetter = nextLetter;
            }

            System.out.println("Generado " + (test + 1) + ": " + generatedWord.toString());
        }

        System.out.println("\n✅ Prueba terminada.");
    }
}
