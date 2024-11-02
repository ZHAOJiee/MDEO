# -*-coding:utf-8 -*-
# 目标求解2*sin(x)+cos(x)最大值
import random
from utils import *
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm


class MDEO(object):

    def __init__(self, method, population_size, budgets, pc, pm, ratio, *graphs):

        self.method = method
        self.population_size = population_size
        self.budgets = budgets
        self.pc = pc
        self.pm = pm
        self.ratio = ratio
        self.graphs = graphs
        self.transfer_number = 30
        self.loop_number = 20
        self.evolution_round = 200


    def generate_population(self, add_edge, delete_edge, budget):
        Population = []
        for i in range(self.population_size):
            # 染色体暂存器
            if budget == 0:
                budget = budget + 1
            ratio = np.random.randint(0, budget)
            add = random.sample(add_edge, ratio)
            delete = random.sample(delete_edge, budget - ratio)  # 采样p的比例的节点作为增边
            temporary = [add, delete, [ratio]]
            temporary = [i for item in temporary for i in item]
            Population.append(temporary)

        return Population
        # 将种群返回，种群是个二维数组，个体和染色体两维

    def single_fitness(self, population, graph, community_old):
        graph_copy = graph.copy()
        ratio = population[-1]
        graph_copy.add_edges(population[:ratio])
        graph_copy.delete_edges(population[ratio:-1])
        community_new = find_communities(graph_copy, self.method)
        objective_1 = confusion_value(community_old, community_new)
        # modularity = math.exp(-community_new.modularity)

        return objective_1

    # return math.log(abs.sum() + 1) * modularity
    # return c_value

    def function(self, population, graph, community_old):
        function1 = []
        for i in range(len(population)):
            pop_fitness = self.single_fitness(population[i], graph, community_old)

            function1.append(pop_fitness)

        # 这里将sin(x)作为目标函数
        return np.array(function1)

    # 3.选择种群中个体适应度最大的个体

    def select(self, population, fitness_value):  # nature selection wrt pop's fitness

        tem_index = random.sample(range(len(population)), self.population_size)
        tem_pop = np.array(population, dtype=object)[tem_index]
        population_select = tem_pop.tolist()

        b = sorted(enumerate(fitness_value), key=lambda x: x[1], reverse=True)
        c = []
        rank = []
        for x in b[0:int(self.population_size * self.ratio)]:
            c.append(x[0])
        elitism = deepcopy(c)
        pop_elitism = np.array(population, dtype=object)[elitism].tolist()
        for x in b[0:self.transfer_number]:
            rank.append(x[0])
        pop_rank = np.array(population, dtype=object)[rank].tolist()

        return population_select, pop_elitism, pop_rank

    def crossover(self, pop):
        new_pop = copy.deepcopy(pop)
        for i, father in enumerate(pop):  # 遍历种群中的每一个个体，将该个体作为父亲
            if np.random.rand() < self.pc:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                mother = pop[np.random.randint(self.population_size)]  # 再种群中选择另一个个体，并将该个体作为母亲
                child1, child2, cross_point = cross(father, mother)
                if len(cross_point) == 0:  # 没交叉
                    continue
                if len(set(child1)) == len(child1):
                    new_pop.append(child1)

                if len(set(child2)) == len(child2):
                    new_pop.append(child2)

        return new_pop

    def mutation(self, population, add_edge, delete_edge):
        # pm是概率阈值
        px = len(population)
        # 求出种群中所有种群/个体的个数
        py = len(population[0]) - 1  # 染色体长度
        # 染色体/个体基因的个数
        for i in range(px):
            mpoint = random.randint(0, py - 1)
            if random.random() < self.pm:
                ratio = population[i][-1]
                pre = copy.deepcopy(population[i])
                #
                if mpoint < ratio:  # 对加边进行变异
                    # 将mpoint个基因进行单点随机变异，变为0或者1
                    add_edge = list(set(add_edge).difference(set(population[i][:ratio])))
                    if len(add_edge) == 0:
                        continue
                    population[i][mpoint] = random.sample(add_edge, 1)
                    population[i][mpoint] = population[i][mpoint][0]
                else:  # 对减边进行变异
                    if len(delete_edge) == 0:
                        continue
                    delete_edge = list(set(delete_edge).difference(set(population[i][ratio:-1])))
                    population[i][mpoint] = random.sample(delete_edge, 1)
                    population[i][mpoint] = population[i][mpoint][0]
                population[i] = pre
        return population

    # 寻找最好的适应度和个体

    def best(self, population, graph, community_old):

        bestindividual = population[0]
        bestfitness = self.single_fitness(bestindividual, graph, community_old)

        return [bestindividual, bestfitness]

    def main(self):

        similarity_matrix = np.load('result/graphs_similarity.npy')
        top3_largest_indices = np.argsort(-similarity_matrix, axis=1)[:, :3]

        print(top3_largest_indices)

        TS = deepcopy(top3_largest_indices)
        mappings = {}

        for i, graph1 in enumerate(self.graphs):
            for j in top3_largest_indices[i]:
                graph2 = self.graphs[j]
                path = 'result/mapping/' + graph2 + '_to_' + graph1 + '.pkl'
                mapping_dict = read_dictionary(path)
                mappings[(j, i)] = mapping_dict

        # Read graphs and find communities
        graphs = [graph_read(graph) for graph in self.graphs]
        communities = [find_communities(graph, self.method) for graph in graphs]

        Fitness_T_average = [[] for _ in range(8)]

        for loop in range(self.loop_number):
            print('The current loop is:', loop)

            cofficient_matrix = filter_matrix(similarity_matrix, top3_largest_indices)
            cofficient_matrix = normalize_matrix(cofficient_matrix)

            # Initialize results_T and Fitness_T lists
            results_T = [[] for _ in range(8)]
            Fitness_T = [[] for _ in range(8)]

            pop = []
            pop_T = []

            population_update = [[None, None, None] for _ in range(8)]
            intersection = [[None, None, None] for _ in range(8)]

            delete_edge = []
            add_edge = []
            for i in range(8):
                # Get edge lists and add pools
                delete_edge.append(graphs[i].get_edgelist())
                add_edge.append(add_pool(graphs[i], 5))

                # Generate populations
                pop.append(self.generate_population(add_edge[i], delete_edge[i], self.budgets[i]))

                # Create deep copies for pop_T
                pop_T.append(deepcopy(pop[i]))

            # Initialize archive, transfer, and index lists
            archive_elitism = [[] for _ in range(8)]
            archive_transfer = [[] for _ in range(8)]
            transfer_index = [[] for _ in range(8)]

            # Initialize add_transfer and delete_transfer lists
            add_transfer = [[] for _ in range(8)]
            delete_transfer = [[] for _ in range(8)]
            fitness_value = [[] for _ in range(8)]
            pop_elitism = [[] for _ in range(8)]
            pop_rank = [[] for _ in range(8)]

            for t in tqdm(range(1, self.evolution_round)):

                for index in range(8):
                    fitness_value[index] = self.function(pop_T[index], graphs[index], communities[index])
                    pop_T[index], pop_elitism[index], pop_rank[index] = self.select(pop_T[index], fitness_value[index])
                    pop_T[index] = self.crossover(pop_T[index])

                    # Perform mutation with appropriate edge set
                    if len(add_transfer[index]) == 0:
                        pop_T[index] = self.mutation(pop_T[index], add_edge[index], delete_edge[index])
                    else:
                        pop_T[index] = self.mutation(pop_T[index], add_transfer[index], delete_transfer[index])

                    # Extend population with elite individuals
                    pop_T[index].extend(pop_elitism[index])

                    # Record best individual and fitness
                    best_individual, best_fitness = self.best(pop_elitism[index], graphs[index], communities[index])
                    results_T[index].append([best_fitness, best_individual])
                    Fitness_T[index].append(best_fitness)

                    archive_elitism[index].append(pop_elitism[index])

                    # Handling transfer logic after certain rounds
                    if t > 10:
                        D1 = Fitness_T[index][-1] - Fitness_T[index][-5]
                        D2 = Fitness_T[index][-5] - Fitness_T[index][-9]
                        if D1 <= D2 and t % 5 == 0:
                            if len(transfer_index[index]) > 1:
                                old_index = transfer_index[index][-1]
                                archive_old = archive_elitism[index][old_index]
                                current_elitism = archive_elitism[index][-1]
                                archive_transfer_set = archive_transfer[index][-1]

                                # Update intersections and coefficient matrix
                                for t_index in range(3):
                                    intersection[index][t_index] = improved_rate(archive_old, current_elitism,
                                                                                 archive_transfer_set[t_index])
                                    cofficient_matrix[index][TS[index][t_index]] += intersection[index][t_index]

                                cofficient_matrix = normalize_matrix(cofficient_matrix)

                                # Calculate ratios and update population
                            ratios = [int(cofficient_matrix[index][t] * self.transfer_number) + 1 for t in TS[index]]

                            for t_index in range(3):
                                population_update[index][t_index] = solution_transfer(graphs[index],
                                                                                      self.budgets[index],
                                                                                      add_edge[index],
                                                                                      delete_edge[index],
                                                                                      pop_rank[TS[index][t_index]][
                                                                                      :ratios[t_index]],
                                                                                      mappings[
                                                                                          (TS[index][t_index], index)])
                                pop_T[index].extend(population_update[index][t_index])

                            # Update transfer information
                            transfer_index[index].append(t)
                            archive_transfer[index].append([population_update[index][0],
                                                            population_update[index][1],
                                                            population_update[index][2]])
                            add_transfer[index], delete_transfer[index] = [], []
                            transfer_pool = sum([population_update[index][i] for i in range(3)], [])
                            for edge_set in transfer_pool:
                                R = edge_set[-1]
                                add_transfer[index].extend(edge_set[0:R])
                                delete_transfer[index].extend(edge_set[R:-1])

            # Calculate and save fitness results for all graphs
            for i in range(8):
                Fitness_T_average[i].append(Fitness_T[i])


if __name__ == '__main__':
    Graphs = ['adjnoun', 'dolphins', 'lesmis', 'Erods', 'polbooks', 'netscience', 'USAir', 'bio_celegans']
    Graph0, Graph1, Graph2, Graph3, Graph4, Graph5, Graph6, Graph7 = Graphs

    Methods = ['fastgreedy']

    for Method in Methods:
        Budgets = [20, 10, 10, 50, 20, 30, 100, 50]
        ga = MDEO(Method, 100,
                 Budgets, 0.5, 0.1, 0.1, Graph0, Graph1, Graph2, Graph3, Graph4, Graph5, Graph6, Graph7)
        ga.main()
