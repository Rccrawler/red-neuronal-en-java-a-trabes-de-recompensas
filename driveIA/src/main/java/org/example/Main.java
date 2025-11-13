package org.example;

public class Main {

    public static void main(String[] args) {

        RLAgent agent = new RLAgent();

        int episodes = 200;

        for (int e = 1; e <= episodes; e++) {

            int x = 0, y = 0; // inicio
            int goalX = 3, goalY = 0;

            int steps = 0;

            while (true) {

                int action = agent.selectAction(x, y);

                int newX = x, newY = y;

                switch (action) {
                    case 0 -> newY--; // UP
                    case 1 -> newY++; // DOWN
                    case 2 -> newX--; // LEFT
                    case 3 -> newX++; // RIGHT
                }

                newX = Math.max(0, Math.min(3, newX));
                newY = Math.max(0, Math.min(3, newY));

                double reward = (newX == goalX && newY == goalY) ? 10 : -0.1;

                agent.learn(x, y, action, reward, newX, newY);

                x = newX;
                y = newY;
                steps++;

                if (reward == 10 || steps > 50) break;
            }

            System.out.println("Episodio " + e + " : pasos = " + steps);
        }

        System.out.println("\n✅ Entrenamiento terminado.");
    }
}

