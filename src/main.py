#By FarukOzderim

import sys
from itertools import chain, combinations
import json
import numpy
from more_itertools import set_partitions
from itertools import chain, combinations
from sympy.utilities.iterables import multiset_permutations

if len(sys.argv) != 3:
    print(f"You need to run with:\n python3 main.py [input] [output]")
    sys.exit(-1)

INPUT = sys.argv[1]
OUTPUT = sys.argv[2]
INT32 = 2 ** 31

# TEST FLAGS
all_min_distance_TEST = False  # for checking find_min_distance_for_all_subsets, open this first for verbose
min_distance_TEST = False  # for checking more details in find_minimum_distance, open this last
main_TEST = False  # for checking main, input reading, open this for input checking, visualizing


# To fix numpy-json compatibility
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class Vehicle:
    """
    Args:
        properties: Array, properties of the vehicle
        number_of_different_combinations: number_of_different_destination_combinations

    Returns:
        Vehicle instance
    """

    def __init__(self, properties, all_subsets, jobs, matrix):
        self.id = properties['id']
        self.start_index = properties['start_index']
        self.capacity = numpy.sum(properties['capacity'])
        self.route_time_distance_list = []
        self.route_order_of_destination_list = []
        self.find_min_distance_for_all_subsets(all_subsets, jobs, matrix)

    def find_min_distance_for_all_subsets(self, all_subsets, jobs, matrix):
        """
        Finds the minimum distance time for all given subsets of job destinations
        Args:
            all_subsets: all subsets of the job destinations
            jobs: all jobs list
            matrix: time distance matrix

        Returns:
            None
        """
        for destinations in all_subsets:

            if all_min_distance_TEST:
                print(
                    f"\ndestinations: {destinations}, vehicle_id: {self.id}, vehicle.start_position:{self.start_index}")

            time_distance, order_of_destinations = self.find_minimum_distance(destinations, jobs, matrix)

            self.route_time_distance_list.append(time_distance)
            self.route_order_of_destination_list.append(order_of_destinations)

        if all_min_distance_TEST:
            print(f"\nself.route_time_distance_list: {self.route_time_distance_list}")
            print(f"self.route_order_of_destination_list: {self.route_order_of_destination_list}")
            print("\n----------------------------------------\n")
            print("\n----------------------------------------\n")

    def find_minimum_distance(self, destinations, jobs, matrix):
        """
            Finds the minimum time-distance for given destinations to the vehicle.

        Args:
            destinations: destinations to cover
            jobs: all job list
            matrix: time distance matrix

        Returns:
            time_distance,
            order_of_destinations
        """
        # if minimum is > than supply return infinite
        destination_no = len(destinations)
        if destination_no == 0:
            return INT32, []

        # sum all demanded cargo
        all_job_ids = numpy.zeros(destination_no, dtype=numpy.int32)  # assume all input are integers
        all_job_locations = numpy.zeros(destination_no, dtype=numpy.int32)
        all_service_times = numpy.zeros(destination_no, dtype=numpy.int32)
        sum_cargo = 0
        sum_service_times = 0
        index = 0

        for destination in destinations:
            job = jobs[destination]
            job_id, job_location_index, job_deliveries, job_service_time = job.values()

            all_job_ids[index] = job_id
            all_job_locations[index] = job_location_index
            all_service_times[index] = job_service_time

            sum_service_times += job_service_time
            sum_cargo += numpy.sum(job_deliveries)
            index += 1

        if min_distance_TEST:
            print("\nmin_distance_TEST$$$")
            print(f"\nnew find_minimum_distance test incoming, vehicle_id:{self.id}:")
            print(f"destinations: {destinations}")
            print(f"sum_cargo: {sum_cargo}")
            print(f"all_job_ids: {all_job_ids}")
            print(f"all_job_locations: {all_job_locations}")
            print(f"all_service_times: {all_service_times}\n")

        if sum_cargo > self.capacity:
            return INT32, []

        all_permutations = list(multiset_permutations(destinations))
        min_order_of_destinations = None
        min_time = INT32

        # Check all permutations for given combination, take the minimum distance_time
        for permutation in all_permutations:
            this_time = self.calculate_distance(self.start_index, permutation, matrix)

            if all_min_distance_TEST:
                print(f"permutation: {permutation}")
                print(f"this_time: {this_time}")
            if min_time > this_time:
                min_time = this_time
                min_order_of_destinations = permutation

        # add service times
        min_time += sum_service_times

        return min_time, min_order_of_destinations

    @staticmethod
    def calculate_distance(start_position, locations, matrix):
        """
        Calculate time distance given locations
        Args:
            start_position: index
            locations: indexes
            matrix: distance between indexes

        Returns: total time distance

        """
        distance = matrix[start_position, locations[0]]

        for i in range(len(locations) - 1):
            distance += matrix[locations[i], locations[i + 1]]
        return distance


class MTSPBruteForce:
    """
    A MTSP Brute Force Solver.

    Args:
        input_path : Array, properties of the worker

    Returns:
        MTSPBruteForce solver
    """

    def __init__(self, input_path):
        vehicles, jobs, matrix = self.read_json(input_path)
        self.vehicles = vehicles
        self.jobs = jobs
        self.matrix = matrix
        self.number_of_vehicles = len(vehicles)
        self.number_of_jobs = len(jobs)
        self.vehicle_list = []
        self.destination_combinations = list(set_partitions(numpy.arange(self.number_of_jobs),
                                                            self.number_of_vehicles))
        self.number_of_combinations = len(self.destination_combinations)
        self.all_destination_subsets = self.generate_all_subsets()

        for vehicle_properties in vehicles:
            self.vehicle_list.append(Vehicle(vehicle_properties, self.all_destination_subsets, self.jobs, self.matrix))

    def run(self):
        return self.find_optimal_solution(self.vehicle_list, self.jobs, self.matrix, self.destination_combinations)

    @staticmethod
    def read_json(path):
        """
        Reads given json file, and returns objects
        Args:
            path, json file path
        Returns:
            vehicles, array of dictionaries
            jobs, array of dictionaries
            matrix, array of arrays
        """
        with open(path, 'r') as myfile:
            data = myfile.read()

        # parse file
        obj = json.loads(data)

        # print obj
        return obj['vehicles'], obj['jobs'], numpy.asarray(obj['matrix'])

    def generate_all_subsets(self):
        """

        Returns:
            all subsets of job destinations
        """
        my_list = numpy.arange(self.number_of_jobs)
        itr = chain.from_iterable(combinations(my_list, n) for n in range(len(my_list) + 1))

        return [list(el) for el in itr]

    def find_optimal_solution(self, vehicles, jobs, matrix, all_combinations):
        """
        Finds the optimal solutions by checking every possible combination
        Args:
            vehicles: all vehicles
            jobs: all jobs
            matrix: all destinations
            all_combinations: all destination distribution combinations

        Returns:
            json: output json object
        """
        print("Finding the optimal solution, it might take some time...")

        output = {}
        routes = {}
        min_time_distance = INT32
        min_routes = {}

        for combination in all_combinations:
            total_time_distance = 0
            total_destination_order_list = []
            routes = {}

            for i in range(len(combination)):
                all_subset_index = self.convert_combination_to_all_destination_subsets_index(combination[i])
                vehicle = vehicles[i]
                vehicle_id = vehicle.id
                vehicle_time = vehicle.route_time_distance_list[all_subset_index]
                vehicle_destination_order = vehicle.route_order_of_destination_list[all_subset_index]

                total_time_distance += vehicle_time
                total_destination_order_list.append(vehicle_destination_order)
                routes[vehicle_id] = {"jobs": vehicle_destination_order, "delivery_duration": vehicle_time}
            if total_time_distance < min_time_distance:
                min_time_distance = total_time_distance
                min_routes = routes

        output["total_delivery_duration"] = min_time_distance
        output["routes"] = min_routes

        json_output = json.dumps(output, cls=NpEncoder, indent=4)
        return json_output

    def convert_combination_to_all_destination_subsets_index(self, combination):
        """
        Returns index of the combination in all_destination_subsets
        Args:
            combination:

        Returns: index
        """
        return self.all_destination_subsets.index(combination)


if __name__ == '__main__':
    my_solver = MTSPBruteForce(INPUT)

    if main_TEST:
        print("\n$$$ Main Test$$$:\n")
        print(f"vehicles: {my_solver.vehicles}")
        print(f"jobs: {my_solver.jobs}")
        print(f"matrix: \n{my_solver.matrix}")
        print(f"number of vehicle-destination distribution combinations: {my_solver.number_of_combinations}")
        print(f"vehicle[0] id : {my_solver.vehicle_list[0].id}")
        print(f"vehicle capacity : {my_solver.vehicle_list[0].capacity}")
        print(f"vehicle start_index : {my_solver.vehicle_list[0].start_index}")
        print(f"all destinations subsets : {my_solver.all_destination_subsets}")
        print(f"all_subsets.index([0, 1]) : {my_solver.all_destination_subsets.index([0, 1])}")
        print(f"all_subsets[4] : {my_solver.all_destination_subsets[4]}")
        print(f"length of all destinations subsets : {len(my_solver.all_destination_subsets)}")
        print("\nTesting the outputs with the matrix:")
        my_matrix = my_solver.matrix
        print(my_matrix[0, 2] + my_matrix[2, 1])
        print(my_solver.generate_all_subsets())

    solution = my_solver.run()
    f = open(OUTPUT, "w")
    f.write(solution)
    f.close()
    print(f"Solution is written to the {OUTPUT}:\n{solution}")
