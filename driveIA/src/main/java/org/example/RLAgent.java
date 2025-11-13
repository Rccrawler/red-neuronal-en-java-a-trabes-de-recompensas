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

public class RLAgent {

    private static final int ACTIONS = 4;  // UP, DOWN, LEFT, RIGHT
    private static final int STATE_SIZE = 2; // x, y
    private static final double GAMMA = 0.99; /*
    Imagina que tienes que elegir entre:
    Opción A: Ganar 1 punto ahora
    Opción B: Ganar 100 puntos en 10 pasos
    Con GAMMA = 0.99, el agente valúa el futuro al 99% de su valor original.
    Fórmula: Valor_Total = Recompensa_Ahora + 0.99 × Recompensa_Futura
    Si GAMMA fuera 0.5, el agente solo cuidaría el presente. Con 0.99, es más inteligente y piensa en el futuro.
    <hr></hr>
    private static final double EPSILON_DECAY = 0.995;
    */
    private static final double EPSILON_DECAY = 0.995; // Epsilon Inicio: epsilon = 1.0 (100% aleatorio) → El agente prueba cosas sin pensarDespués de cada paso: epsilon *= 0.995 → Disminuye un poco Final: epsilon ≈ 0.0 (casi 0%) → El agente solo usa lo que aprendió
    private static final double LEARNING_RATE = 0.01;
    private double epsilon = 1.0; // belocidad de aprendizage
    private Random random = new Random();

    private MultiLayerNetwork model;

    public RLAgent() {

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .updater(new Adam(LEARNING_RATE))
                .list()
                .layer(new DenseLayer.Builder().nIn(STATE_SIZE).nOut(16)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nOut(16)
                        .activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nOut(ACTIONS).build())
                .build();

        model = new MultiLayerNetwork(config);
        model.init();
    }

    // Decide acción usando epsilon-greedy
    public int selectAction(int x, int y) {
        if (random.nextDouble() < epsilon) {
            return random.nextInt(ACTIONS);
        }

        INDArray input = Nd4j.create(new double[][]{{x, y}}); // Corregido: crear como matriz
        INDArray output = model.output(input);
        return Nd4j.argMax(output, 1).getInt(0);
    }

    public void learn(int x, int y, int action, double reward, int nextX, int nextY) {

        INDArray input = Nd4j.create(new double[][]{{x, y}}); // Corregido: crear como matriz
        INDArray target = model.output(input).dup();

        INDArray nextInput = Nd4j.create(new double[][]{{nextX, nextY}}); // Corregido: crear como matriz
        INDArray futureQ = model.output(nextInput);

        double qUpdated = reward + GAMMA * futureQ.maxNumber().doubleValue();
        target.putScalar(action, qUpdated);

        model.fit(input, target);
        epsilon *= EPSILON_DECAY;
    }
}

