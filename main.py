import random
from enum import Enum
import yaml

class RandomNumberGenerator:
    def __init__(self, seed, a, c, m):
        self.seed = seed
        self.a = a
        self.c = c
        self.m = m

    def next(self):
        self.seed = (self.a * self.seed + self.c) % self.m
        return self.seed / self.m


class EventType(Enum):
    ARRIVAL = "CH1"
    DEPARTURE = "SA2"
    PASSAGE = "P12"


class Event:
    def __init__(self, event_type: EventType, time, source_queue=None, destination_queue=None):
        self.type = event_type
        self.time = time
        self.source_queue = source_queue
        self.destination_queue = destination_queue


class Queue:
    def __init__(self, arrival_interval=[], service_interval=[], num_servers=0, capacity=0, is_infinite=True):
        self.arrival_interval = arrival_interval
        self.service_interval = service_interval
        self.num_servers = num_servers
        self.capacity = capacity
        self.is_infinite = is_infinite
        self.states = []
        self.weights = None
        self.losses = 0
        self.population = 0

def print_results(queues, global_time):
    for i, queue in enumerate(queues):
        print(f"Queue {i}:  |-> Population: {queue.population}, ||-> Losses: {queue.losses}, |||-> States: {queue.states}")
    print("\n => Total time: ",global_time )


class Simulator:
    def __init__(self, config_file_path, seed):
        self.scheduler = []
        self.global_time = 0.0
        self.a = 32453245
        self.c = 345617645733753
        self.M = 2**64
        self.seed = seed
        self.initialize_queue_network(config_file_path)
        self.random_generator = RandomNumberGenerator(self.seed, self.a, self.c, self.M)

    def execute(self):
        start_time = self.global_time
        self.arrival(self.queues[0], self.first_arrival)

        while self.iterations > 0:
            self.scheduler.sort(key=lambda x: x.time)
            next_event = self.scheduler.pop(0)

            if next_event.type == EventType.ARRIVAL:
                self.arrival(self.queues[0], next_event.time)
            elif next_event.type == EventType.PASSAGE:
                self.passage(next_event)
            else:
                self.departure(next_event)

        end_time = self.global_time
        total_time_spent = end_time - start_time
        print(f"Total time spent in simulation: {total_time_spent:.4f} units")

        for queue in self.queues:
            queue.states = [round(time, 4) for time in queue.states]

        return self.queues

    def load_yaml(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def initialize_queue_network(self, config_file_path):
        config = self.load_yaml(config_file_path)
        arrivals = config['parameters']['arrivals']
        queues_data = config['parameters']['queues']
        network = config['parameters']['network']
        self.iterations = config['parameters']['rndnumbersPerSeed']
        self.first_arrival = arrivals['Q1']

        num_queues = len(queues_data)
        weight_matrix = [[0] * num_queues for _ in range(num_queues)]

        queue_names = list(queues_data.keys())
        queues_data = list(queues_data.values())

        for connection in network:
            source_index = queue_names.index(connection['source'])
            target_index = queue_names.index(connection['target'])
            weight_matrix[source_index][target_index] = connection['probability']

        for i in range(num_queues):
            row_sum = sum(weight_matrix[i])
            weight_matrix[i][i] = 1 - row_sum

        self.queues = [Queue() for _ in range(len(queues_data))]

        for i in range(num_queues):
            self.queues[i].arrival_interval = [queues_data[i].get('minArrival', 0.0), queues_data[i].get('maxArrival', 0.0)]
            self.queues[i].service_interval = [queues_data[i]['minService'], queues_data[i]['maxService']]
            self.queues[i].capacity = queues_data[i].get('capacity', 0)
            self.queues[i].infinite = self.queues[i].capacity == 0
            self.queues[i].num_servers = queues_data[i]['servers']
            self.queues[i].states = [0.0] * (self.queues[i].capacity + 1)
            self.queues[i].weights = weight_matrix[i]

    def arrival(self, queue, time):
        self.update_queue_times(time)

        if queue.infinite and (queue.population == queue.capacity):
            queue.states.append(0)
            queue.capacity += 1

        if queue.population < queue.capacity:
            queue.population += 1

            if queue.population <= queue.num_servers:
                source_index = self.queues.index(queue)
                destination_index = self.queues.index(random.choices(self.queues, queue.weights)[0])
                self.schedule_passage(source_index, destination_index)
        else:
            queue.losses += 1
        self.schedule_arrival()

    def departure(self, event):
        self.update_queue_times(event.time)
        source_queue = self.queues[event.source_queue]

        source_queue.population -= 1
        if source_queue.population >= source_queue.num_servers:
            destination_index = self.queues.index(random.choices(self.queues, source_queue.weights)[0])

            if event.source_queue == destination_index:
                self.schedule_departure(event.source_queue)
            else:
                self.schedule_passage(event.source_queue, destination_index)

    def passage(self, event):
        self.update_queue_times(event.time)
        source_queue = self.queues[event.source_queue]
        destination_queue = self.queues[event.destination_queue]

        source_queue.population -= 1

        if source_queue.population >= source_queue.num_servers:
            source_index = self.queues.index(source_queue)
            destination_index = self.queues.index(random.choices(self.queues, source_queue.weights)[0])

            if source_index == destination_index:
                self.schedule_departure(source_index)
            else:
                self.schedule_passage(source_index, destination_index)

        if destination_queue.population >= destination_queue.capacity and destination_queue.infinite:
            destination_queue.states.append(0)
            destination_queue.capacity += 1
        
        if destination_queue.population < destination_queue.capacity:
            destination_queue.population += 1

            if destination_queue.population <= destination_queue.num_servers:
                source_index = self.queues.index(destination_queue)
                destination_index = self.queues.index(random.choices(self.queues, destination_queue.weights)[0])

                if source_index == destination_index:
                    self.schedule_departure(source_index)
                else:
                    self.schedule_passage(source_index, destination_index)
        else:
            destination_queue.losses += 1

    def schedule_arrival(self):
        arrival_event = Event(
            EventType.ARRIVAL,
            self.global_time + self.random_draw(self.queues[0].arrival_interval[0], self.queues[0].arrival_interval[1])
        )
        self.scheduler.append(arrival_event)

    def schedule_departure(self, source_queue_index):
        departure_event = Event(
            EventType.DEPARTURE,
            self.global_time + self.random_draw(self.queues[source_queue_index].service_interval[0], self.queues[source_queue_index].service_interval[1]),
            source_queue=source_queue_index
        )
        self.scheduler.append(departure_event)

    def schedule_passage(self, source_queue_index, destination_queue_index):
        passage_event = Event(
            EventType.PASSAGE,
            self.global_time + self.random_draw(self.queues[source_queue_index].service_interval[0], self.queues[source_queue_index].service_interval[1]),
            source_queue=source_queue_index,
            destination_queue=destination_queue_index
        )
        self.scheduler.append(passage_event)

    def update_queue_times(self, time):
        for queue in self.queues:
            queue.states[queue.population] += (time - self.global_time)
        self.global_time = time

    def random_draw(self, min_val, max_val):
        self.iterations -= 1
        return (max_val - min_val) * self.random_generator.next() + min_val
    def __init__(self, config_file_path, seed):
        self.scheduler = []
        self.global_time = 0.0
        self.a = 32453245
        self.c = 345617645733753
        self.M = 2**64
        self.seed = seed
        self.initialize_queue_network(config_file_path)
        self.random_generator = RandomNumberGenerator(self.seed, self.a, self.c, self.M)

    def execute(self):
        self.arrival(self.queues[0], self.first_arrival)

        while self.iterations > 0:
            self.scheduler.sort(key=lambda x: x.time)
            next_event = self.scheduler.pop(0)

            if next_event.type == EventType.ARRIVAL:
                self.arrival(self.queues[0], next_event.time)
            elif next_event.type == EventType.PASSAGE:
                self.passage(next_event)
            else:
                self.departure(next_event)

        for queue in self.queues:
            queue.states = [round(time, 4) for time in queue.states]

        return self.queues

    def load_yaml(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def initialize_queue_network(self, config_file_path):
        config = self.load_yaml(config_file_path)
        arrivals = config['parameters']['arrivals']
        queues_data = config['parameters']['queues']
        network = config['parameters']['network']
        self.iterations = config['parameters']['rndnumbersPerSeed']
        self.first_arrival = arrivals['Q1']

        num_queues = len(queues_data)
        weight_matrix = [[0] * num_queues for _ in range(num_queues)]

        queue_names = list(queues_data.keys())
        queues_data = list(queues_data.values())

        for connection in network:
            source_index = queue_names.index(connection['source'])
            target_index = queue_names.index(connection['target'])
            weight_matrix[source_index][target_index] = connection['probability']

        for i in range(num_queues):
            row_sum = sum(weight_matrix[i])
            weight_matrix[i][i] = 1 - row_sum

        self.queues = [Queue() for _ in range(len(queues_data))]

        for i in range(num_queues):
            self.queues[i].arrival_interval = [queues_data[i].get('minArrival', 0.0), queues_data[i].get('maxArrival', 0.0)]
            self.queues[i].service_interval = [queues_data[i]['minService'], queues_data[i]['maxService']]
            self.queues[i].capacity = queues_data[i].get('capacity', 0)
            self.queues[i].infinite = self.queues[i].capacity == 0
            self.queues[i].num_servers = queues_data[i]['servers']
            self.queues[i].states = [0.0] * (self.queues[i].capacity + 1)
            self.queues[i].weights = weight_matrix[i]

    def arrival(self, queue, time):
        self.update_queue_times(time)

        if queue.infinite and (queue.population == queue.capacity):
            queue.states.append(0)
            queue.capacity += 1

        if queue.population < queue.capacity:
            queue.population += 1

            if queue.population <= queue.num_servers:
                source_index = self.queues.index(queue)
                destination_index = self.queues.index(random.choices(self.queues, queue.weights)[0])
                self.schedule_passage(source_index, destination_index)
        else:
            queue.losses += 1
        self.schedule_arrival()

    def departure(self, event):
        self.update_queue_times(event.time)
        source_queue = self.queues[event.source_queue]

        source_queue.population -= 1
        if source_queue.population >= source_queue.num_servers:
            destination_index = self.queues.index(random.choices(self.queues, source_queue.weights)[0])

            if event.source_queue == destination_index:
                self.schedule_departure(event.source_queue)
            else:
                self.schedule_passage(event.source_queue, destination_index)

    def passage(self, event):
        self.update_queue_times(event.time)
        source_queue = self.queues[event.source_queue]
        destination_queue = self.queues[event.destination_queue]

        source_queue.population -= 1

        if source_queue.population >= source_queue.num_servers:
            source_index = self.queues.index(source_queue)
            destination_index = self.queues.index(random.choices(self.queues, source_queue.weights)[0])

            if source_index == destination_index:
                self.schedule_departure(source_index)
            else:
                self.schedule_passage(source_index, destination_index)

        if destination_queue.population >= destination_queue.capacity and destination_queue.infinite:
            destination_queue.states.append(0)
            destination_queue.capacity += 1
        
        if destination_queue.population < destination_queue.capacity:
            destination_queue.population += 1

            if destination_queue.population <= destination_queue.num_servers:
                source_index = self.queues.index(destination_queue)
                destination_index = self.queues.index(random.choices(self.queues, destination_queue.weights)[0])

                if source_index == destination_index:
                    self.schedule_departure(source_index)
                else:
                    self.schedule_passage(source_index, destination_index)
        else:
            destination_queue.losses += 1

    def schedule_arrival(self):
        arrival_event = Event(
            EventType.ARRIVAL,
            self.global_time + self.random_draw(self.queues[0].arrival_interval[0], self.queues[0].arrival_interval[1])
        )
        self.scheduler.append(arrival_event)

    def schedule_departure(self, source_queue_index):
        departure_event = Event(
            EventType.DEPARTURE,
            self.global_time + self.random_draw(self.queues[source_queue_index].service_interval[0], self.queues[source_queue_index].service_interval[1]),
            source_queue=source_queue_index
        )
        self.scheduler.append(departure_event)

    def schedule_passage(self, source_queue_index, destination_queue_index):
        passage_event = Event(
            EventType.PASSAGE,
            self.global_time + self.random_draw(self.queues[source_queue_index].service_interval[0], self.queues[source_queue_index].service_interval[1]),
            source_queue=source_queue_index,
            destination_queue=destination_queue_index
        )
        self.scheduler.append(passage_event)

    def update_queue_times(self, time):
        for queue in self.queues:
            queue.states[queue.population] += (time - self.global_time)
        self.global_time = time

    def random_draw(self, min_val, max_val):
        self.iterations -= 1
        return (max_val - min_val) * self.random_generator.next() + min_val

if __name__ == '__main__':
    simulator = Simulator('config.yml', 76527852567925689254689425)
    queues = simulator.execute()
    print_results(queues, simulator.global_time)