import logging
from enum import Enum, IntEnum

import matplotlib.pyplot as plt
import numpy as np


class Cell(IntEnum):
    EMPTY = 0  # indica le celle libere
    OCCUPIED = 1  # indica le celle occupate 
    CURRENT = 2  # indica la cella dove si trova l'agente


class Action(IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3


class Render(Enum):
    NOTHING = 0
    TRAINING = 1
    MOVES = 2


class Status(Enum):
    WIN = 0
    LOSE = 1
    PLAYING = 2


class Maze:
    """ Un labirinto con ostacoli. Un agente è posizionato nella cella di start e deve trovare l'uscita muovendosi attraverso la mappa.

        Il layout del labirinto e le regole di come muoversi attraverso fanno parte dell'environment.
        Un agente è posizionato nella cella di start.L'agente sceglie azioni (move left/right/up/down) per raggiungere la cella target.
        Ogni azione risulta in un premio o una penalità che vengono accumulate durante la partita. ogni mossa dà una piccola
        penalità (-0.05), ritornare in una cella già visitata una penalità più grande (-0.25) e andare incontro ad un muro 
        una penalità maggiore (-0.75). Il premio (+10.0) viene concesso quando l'agente raggiunge l'uscita. La
        partita raggiunge sempre uno stato finale; indipendentemente dal fatto che l'agente vinca o perda.
        Ovviamente raggiungere l'uscita significa vincere, ma se le penalità che l'agente accumula durante la partita superano
        un certo valore, si assume che l'agente si sta muovendo a caso e perde automaticamente.

        A note on cell coordinates:
        The cells in the maze are stored as (col, row) or (x, y) tuples. (0, 0) is the upper left corner of the maze.
        This way of storing coordinates is in line with what matplotlib's plot() function expects as inputs. The maze
        itself is stored as a 2D numpy array so cells are accessed via [row, col]. To convert a (col, row) tuple
        to (row, col) use (col, row)[::-1]
    """
    actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN]  # tutte le possibili azioni

    reward_exit = 10.0  # premio per aver raggiunto la cella target
    penalty_move = -0.05  # costo per muoversi in una cella nuova che non è quella target
    penalty_visited = -0.25  # costo per muoversi in una cella già visitata
    penalty_impossible_move = -0.75  # penalità per aver provato a muoversi in una cella occupata

    def __init__(self, maze, start_cell=(8, 9), alpha=None, bravo=None, charlie=None, delta=None):
        """ Crea un nuovo labirinto.

            :param numpy.array maze: array 2D contenente spazi vuoti (= 0) e spazi occupati dai muri (= 1)
            :param tuple start_cell: cella iniziale per l'agente nella mappa (opzionale, altrimenti inizia nell'angolo in alto a sinistra)
            :param tuple food_cell: la cella finale che l'agente deve raggiungere (opzionale, altrimenti nell'angolo a destra)
        """
        self.maze = maze

        self.__minimum_reward = -0.5 * self.maze.size  # ferma il gioco se il reward accumulato è troppo basso

        nrows, ncols = self.maze.shape
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == Cell.EMPTY]
        self.__alpha = (ncols - 2, nrows - 2) if alpha is None else alpha
        self.__bravo = (1, nrows - 2) if bravo is None else bravo
        self.__charlie = (ncols - 2, 1) if charlie is None else charlie
        self.__delta = (1, 1) if delta is None else delta
        self.empty.remove(self.__alpha)
        self.empty.remove(self.__bravo)
        self.empty.remove(self.__charlie)
        self.empty.remove(self.__delta)



        # controllo per layout di mappe errato
        if self.__alpha not in self.cells:
            raise Exception("Error: la cella di arrivo {} non è all'interno della mappa".format(self.__alpha))
        if self.maze[self.__alpha[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: la cella di arrivo {} non è libera".format(self.__alpha))

        if self.__bravo not in self.cells:
            raise Exception("Error: la cella di arrivo {} non è all'interno della mappa".format(self.__bravo))
        if self.maze[self.__bravo[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: la cella di arrivo {} non è libera".format(self.__bravo))

        if self.__charlie not in self.cells:
            raise Exception("Error: la cella di arrivo {} non è all'interno della mappa".format(self.__charlie))
        if self.maze[self.__charlie[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: la cella di arrivo {} non è libera".format(self.__charlie))
        
        if self.__delta not in self.cells:
            raise Exception("Error: la cella di arrivo {} non è all'interno della mappa".format(self.__delta))
        if self.maze[self.__delta[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: la cella di arrivo {} non è libera".format(self.__delta))

        # variabili per il rendering utilizzando Matplotlib
        self.__render = Render.NOTHING  # cosa renderizzare
        self.__ax1 = None  # asse per renderizzare le mosse
        self.__ax2 = None  # asse per renderizzare la mossa migliore per cella

        self.reset(start_cell)

    def reset(self, start_cell=(8, 9)):
        """ Reset del labirinto al proprio stato iniziale e posiziona l'agente nella cella iniziale.

            :param tuple start_cell: qui l'agente inizia la sua avventura all'interno del labirinto (opzionale, altrimenti nell'angolo in alto a sinistra)
            :return: nuovo stato dopo il reset
        """
        if start_cell not in self.cells:
            raise Exception("Error: la cella iniziale a {} non è dentro la mappa".format(start_cell))
        if self.maze[start_cell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: la cella iniziale a {} non è libera".format(start_cell))
        if start_cell == self.__alpha:
            raise Exception("Error: la cella iniziale non può essere uguale ad una cella finale {}".format(start_cell))
        if start_cell == self.__bravo:
            raise Exception("Error: la cella iniziale non può essere uguale ad una cella finale {}".format(start_cell))
        if start_cell == self.__charlie:
            raise Exception("Error: la cella iniziale non può essere uguale ad una cella finale {}".format(start_cell))
        if start_cell == self.__delta:
            raise Exception("Error: la cella iniziale non può essere uguale ad una cella finale {}".format(start_cell))

        self.__previous_cell = self.__current_cell = start_cell
        self.__total_reward = 0.0  # punteggio realizzato
        self.__visited = set()  # un set() imagazzina solo valori unici

        if self.__render in (Render.TRAINING, Render.MOVES):
            # renderizza la mappa
            nrows, ncols = self.maze.shape
            self.__ax1.clear()
            self.__ax1.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax1.set_xticklabels([])
            self.__ax1.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax1.set_yticklabels([])
            self.__ax1.grid(True)
            self.__ax1.plot(*self.__current_cell, "rs", markersize=16)  # la cella iniziale è contrasegnata in rosso
            self.__ax1.text(*self.__current_cell, "P", ha="center", va="center", color="white", fontsize = 10)
            self.__ax1.plot(*self.__alpha, "gs", markersize=16)  # la fine è contrassegnata in verde
            self.__ax1.text(*self.__alpha, "A", ha="center", va="center", color="white", fontsize = 10)
            self.__ax1.plot(*self.__bravo, "ys", markersize=16)  # la fine è contrassegnata in verde
            self.__ax1.text(*self.__bravo, "B", ha="center", va="center", color="white", fontsize = 10)
            self.__ax1.plot(*self.__charlie, "cs", markersize=16)  # la fine è contrassegnata in verde
            self.__ax1.text(*self.__charlie, "C", ha="center", va="center", color="white", fontsize = 10)
            self.__ax1.plot(*self.__delta, "ms", markersize=16)  # la fine è contrassegnata in verde
            self.__ax1.text(*self.__delta, "D", ha="center", va="center", color="white", fontsize = 10)
            self.__ax1.imshow(self.maze, cmap="binary")
            self.__ax1.get_figure().canvas.draw()
            self.__ax1.get_figure().canvas.flush_events()

        return self.__observe()

    def __draw(self):
        """ Disegna una linea dalle celle già visitate fino a quella corrente. """
        self.__ax1.plot(*zip(*[self.__previous_cell, self.__current_cell]), "bo-")  # le celle visitate sono segnate in blu
        self.__ax1.plot(*self.__current_cell, "ro")  # la cella dove si trova e segnata in rosso
        self.__ax1.get_figure().canvas.draw()
        self.__ax1.get_figure().canvas.flush_events()

    def render(self, content=Render.NOTHING):
        """ memorizza quello che verra renderizzato durante il gioco e/o il training.

            :param Render content: NOTHING, TRAINING, MOVES
        """
        self.__render = content

        if self.__render == Render.NOTHING:
            if self.__ax1:
                self.__ax1.get_figure().close()
                self.__ax1 = None
            if self.__ax2:
                self.__ax2.get_figure().close()
                self.__ax2 = None
        if self.__render == Render.TRAINING:
            if self.__ax2 is None:
                fig, self.__ax2 = plt.subplots(1, 1, tight_layout=True)
                #fig.canvas.set_window_title("Best move")
                self.__ax2.set_axis_off()
                self.render_q(None)
        if self.__render in (Render.MOVES, Render.TRAINING):
            if self.__ax1 is None:
                fig, self.__ax1 = plt.subplots(1, 1, tight_layout=True)
                #fig.canvas.set_window_title("Pacman")

        plt.show(block=False)

    def step(self, action):
        """ Muove l'agente in base a 'action' e ritorna il nuovo stato, premio e stato del gioco.

            :param Action action: l'agente si muoverà in questa direzione
            :return: state, reward, status
        """
        reward = self.__execute(action)
        self.__total_reward += reward
        status = self.__status()
        state = self.__observe()
        logging.debug("action: {:10s} | reward: {: .2f} | status: {}".format(Action(action).name, reward, status))
        return state, reward, status

    def __execute(self, action):
        """ esegue l'azione e aggiorna il reward.

            :param Action action: direzione nella quale l'agente si muove
            :return float: premio o penalità che risulta dal movimento
        """
        possible_actions = self.__possible_actions(self.__current_cell)

        if not possible_actions:
            reward = self.__minimum_reward - 1  # se non può muoversi da nessuna parte forza la fine del gioco
        elif action in possible_actions:
            col, row = self.__current_cell
            if action == Action.MOVE_LEFT:
                col -= 1
            elif action == Action.MOVE_UP:
                row -= 1
            if action == Action.MOVE_RIGHT:
                col += 1
            elif action == Action.MOVE_DOWN:
                row += 1

            self.__previous_cell = self.__current_cell
            self.__current_cell = (col, row)

            if self.__render != Render.NOTHING:
                self.__draw()

            if self.__current_cell == self.__alpha:
                reward = Maze.reward_exit  # reward massimo per aver raggiunto la cella target
            elif self.__current_cell in self.__visited:
                reward = Maze.penalty_visited  # penalità per passare in una cella già visitata
            else:
                reward = Maze.penalty_move  # costo del movimento

            self.__visited.add(self.__current_cell)
        else:
            reward = Maze.penalty_impossible_move  # penalità per aver provato a muoversi in una cella occupata

        return reward

    def __possible_actions(self, cell=None):
        """ crea una lista di possibili azioni da 'cell', evitando i limiti della mappa e i muri.

            :param tuple cell: posizione dell'agente (opzionale, cella corrente)
            :return list: tutte le possibili azioni
        """
        if cell is None:
            col, row = self.__current_cell
        else:
            col, row = cell

        possible_actions = Maze.actions.copy()  # inizia che sono tutte permesse

        #  da qui iniza a rimuovere
        nrows, ncols = self.maze.shape
        if row == 0 or (row > 0 and self.maze[row - 1, col] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_UP)
        if row == nrows - 1 or (row < nrows - 1 and self.maze[row + 1, col] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_DOWN)

        if col == 0 or (col > 0 and self.maze[row, col - 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_LEFT)
        if col == ncols - 1 or (col < ncols - 1 and self.maze[row, col + 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_RIGHT)

        return possible_actions

    def __status(self):
        """ ritorna lo stato della partita.

            :return Status: stato corrente della partita (WIN, LOSE, PLAYING)
        """
        if self.__current_cell == self.__alpha or self.__current_cell == self.__bravo or self.__current_cell == self.__charlie or self.__current_cell == self.__delta:
            return Status.WIN

        if self.__total_reward < self.__minimum_reward:  # forza la chiusura della partita se reward troppo basso
            return Status.LOSE

        return Status.PLAYING

    def __observe(self):
        """ Return the state of the maze - in this game the agents current location.

            :return numpy.array [1][2]: posizione attuale dell'agente
        """
        return np.array([[*self.__current_cell]])

    def play(self, model, start_cell=(0, 0)):
        """ Play a single game, choosing the next move based a prediction from 'model'.

            :param class AbstractModel model: the prediction model to use
            :param tuple start_cell: agents initial cell (optional, else upper left)
            :return Status: WIN, LOSE
        """
        self.reset(start_cell)

        state = self.__observe()

        while True:
            action = model.predict(state=state)
            state, reward, status = self.step(action)
            if status in (Status.WIN, Status.LOSE):
                return status

    def check_win_all(self, model):
        """ controlla se l'agente vince indipendentemente da dove inizia. """
        previous = self.__render
        self.__render = Render.NOTHING  # avoid rendering anything during execution of the check games

        win = 0
        lose = 0

        for cell in self.empty:
            if self.play(model, cell) == Status.WIN:
                win += 1
            else:
                lose += 1

        self.__render = previous  # restore previous rendering setting

        logging.info("won: {} | lost: {} | win rate: {:.5f}".format(win, lose, win / (win + lose)))

        result = True if lose == 0 else False

        return result, win / (win + lose)

    def render_q(self, model):
        """ Render the recommended action(s) for each cell as provided by 'model'.

        :param class AbstractModel model: the prediction model to use
        """

        def clip(n):
            return max(min(1, n), 0)

        if self.__render == Render.TRAINING:
            nrows, ncols = self.maze.shape

            self.__ax2.clear()
            self.__ax2.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax2.set_xticklabels([])
            self.__ax2.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax2.set_yticklabels([])
            self.__ax2.grid(True)
            self.__ax2.plot(*self.__alpha, "gs", markersize=16)  # la fine è contrassegnata in verde
            self.__ax2.text(*self.__alpha, "A", ha="center", va="center", color="white", fontsize = 10)
            self.__ax2.plot(*self.__bravo, "ys", markersize=16)  # la fine è contrassegnata in verde
            self.__ax2.text(*self.__bravo, "B", ha="center", va="center", color="white", fontsize = 10)
            self.__ax2.plot(*self.__charlie, "cs", markersize=16)  # la fine è contrassegnata in verde
            self.__ax2.text(*self.__charlie, "C", ha="center", va="center", color="white", fontsize = 10)
            self.__ax2.plot(*self.__delta, "ms", markersize=16)  # la fine è contrassegnata in verde
            self.__ax2.text(*self.__delta, "D", ha="center", va="center", color="white", fontsize = 10)

            for cell in self.empty:
                q = model.q(cell) if model is not None else [0, 0, 0, 0]
                a = np.nonzero(q == np.max(q))[0]

                for action in a:
                    dx = 0
                    dy = 0
                    if action == Action.MOVE_LEFT:
                        dx = -0.2
                    if action == Action.MOVE_RIGHT:
                        dx = +0.2
                    if action == Action.MOVE_UP:
                        dy = -0.2
                    if action == Action.MOVE_DOWN:
                        dy = 0.2

                    # color (from red to green) represents the certainty of the preferred action(s)
                    maxv = 1
                    minv = -1
                    color = clip((q[action] - minv) / (maxv - minv))  # normalize in [-1, 1]

                    self.__ax2.arrow(*cell, dx, dy, color=(1 - color, color, 0), head_width=0.2, head_length=0.1)

            self.__ax2.imshow(self.maze, cmap="binary")
            self.__ax2.get_figure().canvas.draw()
