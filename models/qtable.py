import logging
import random
from datetime import datetime

import numpy as np

from environment import Status
from models import AbstractModel


class QTableModel(AbstractModel):
    """ 
    modello Q-learning

    Per ogni stato (nel nostro caso è identificato dalla posizione dell'agente) il valore di ogni azione è memorizzato in una tabella.
    La chiave per questa tabella è (stato + azione). Inizialmente tutti i valori sono 0. Durante le partite di allenamento
    dopo ogni mossa il valore nella tabella è agiiornato basandosi sul premio guadagnato dopo aver effettuato il movimento. L'allenamento
    finisce dopo un determinato numero di partite, o prima se raggiunge un determinato criterio (nel nostro caso un win-rate approssimato al 100%).
    """
    default_check_convergence_every = 5  # controllo del criterio ogni # episodi

    def __init__(self, game, **kwargs):
        """ 
        Crea un modello di previsone per partita.

        :param class Maze game: Maze game object
        :param kwargs: model dependent init parameters
        """
        super().__init__(game, name="QTableModel", **kwargs)
        self.Q = dict()  # Tabella per combinazioni (stato, azione)

    def train(self, stop_at_convergence=False, **kwargs):
        """ 
        Allenamento del modello.

            :param stop_at_convergence: ferma l'allenamento quando raggiunge il criterio

            Hyperparameters:
            :keyword float discount: (gamma) determina l'importanza delle ricompense future (0 = per niente, 1 = soltanto)
            :keyword float exploration_rate: (epsilon) 0 = tasso di esplorazione (0 = per niente, 1 = soltanto)
            :keyword float exploration_decay: riduzione dell'exploration rate dopo ogni step randomico (<= 1, 1 = per niente)
            :keyword float learning_rate: (alpha) tasso di apprendimento (0 = per niente, 1 = soltanto)
            :keyword int episodes: numero di partite di allenamento
            :return int, datetime: numero di episodi di allenamento, tempo totale speso
        """
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # % riduzione per step = 100 - exploration decay
        learning_rate = kwargs.get("learning_rate", 0.10)
        episodes = max(kwargs.get("episodes", 1000), 1)
        check_convergence_every = kwargs.get("check_convergence_every", self.default_check_convergence_every)

        # variables for reporting purposes
        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []

        start_list = list()
        start_time = datetime.now()

        # l'allenamento inizia qui
        for episode in range(1, episodes + 1):
            # ottimizzazione: si assicura di iniziare da ogni cella
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)

            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())  # cambia np.ndarray a tupla così che può essere usato come chiave di un dizionario (dict)

            while True:
                # sceglie l'azione epsilon greedy (off-policy, invece di usare solamente la policy imparata)
                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                else:
                    action = self.predict(state)

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())

                cumulative_reward += reward

                if (state, action) not in self.Q.keys():  # si assicura che il valore (stato, azione) esista per evitare un KeyError
                    self.Q[(state, action)] = 0.0

                max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.actions])

                self.Q[(state, action)] += learning_rate * (reward + discount * max_next_Q - self.Q[(state, action)])

                if status in (Status.WIN, Status.LOSE):  # stato finale raggiunto, ferma l'episodio di allenamento
                    break

                state = next_state

                self.environment.render_q(self)

            cumulative_reward_history.append(cumulative_reward)

            logging.info("episode: {:d}/{:d} | status: {:4s} | e: {:.5f}"
                         .format(episode, episodes, status.name, exploration_rate))

            if episode % check_convergence_every == 0:
                # controlla se il modello corrente vince a prescindere dalla cella iniziale
                # è possibile solo se il numero di celle è finito
                w_all, win_rate = self.environment.check_win_all(self)
                win_history.append((episode, win_rate))
                if w_all is True and stop_at_convergence is True:
                    logging.info("won from all start cells, stop learning")
                    break

            exploration_rate *= exploration_decay  # esplora di meno con l'avanzare dell'allenamento

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time

    def q(self, state):
        """ Ottiene q values per tutte le azioni per un determinato stato. """
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])

    def predict(self, state):
        """ 
        Policy: sceglie l'azione con il valore più alto dalla Q-table.
        La scelta è randomica se più azioni hanno lo stesso valore (il più alto).

            :param np.ndarray state: game state
            :return int: azione selezionata
        """
        q = self.q(state)

        logging.debug("q[] = {}".format(q))

        actions = np.nonzero(q == np.max(q))[0]  # ottiene l'indice dell'azione(s)con il valore piùù alto
        return random.choice(actions)
