import random as rd
import cv2
import math
import chart
import map

MAX_GENE = 1000 # 최대 반복
POP_SIZE = 8 # 염색체 개수
MUT_RATE = 0.13 # 돌연변이 확률


'''
genes = 1차원 배열[0,1,2,3,4,5,6,7,8] = 방문 순서 -> 0~8 순서로 순회하고 0으로 돌아옴
'''

distance = [] # 장소간 거리
spots = [] # 점이 찍힌 장소들

class Spot: # 한 장소
    num = 0
    def __init__(self, x, y):
        self.id = Spot.num
        self.x = x
        self.y = y
        Spot.num += 1

    def distanceTo(self, spot): # 다른 도시간 거리 구하기
        if spot.id == self.id:
            return 0
        xDistance = abs(self.x - spot.x)
        yDistance = abs(self.y - spot.y)
        return math.sqrt((xDistance*xDistance) + (yDistance*yDistance)) / 2 # 100픽셀이 50미터라 / 2


class Chromosome: # 염색체
    def __init__(self, g=[]): # 생성자
        self.genes = g.copy()
        self.fitness = 0

        if self.genes.__len__() == 0:
            self.genes = [*range(len(spots))]
            rd.shuffle(self.genes) # [1,2,3,4,5,6,7,8]을 무작위로 순서변경
        self.cal_fitness()

    def getFitness(self):
        return self.fitness

    def cal_fitness(self): # 적합도 계산
        self.fitness = 0
    
        for i in range(len(spots) - 1): # 각 염색체에 해당하는 id값을 가진 장소간의 거리 더하기
            self.fitness += distance[self.genes[i]][self.genes[i+1]]
        self.fitness += distance[self.genes[0]][self.genes[-1]]
        return self.fitness


def print_p(pop): # 출력
    for i, x in enumerate(pop):
        print(f"염색체 # {i} = {x.genes} 적합도 = {x.getFitness()}")
    print()

def select(pop): # 부모 선택
    # 적합도에 비례해 선택하면 안좋은 개체를 선택할 확률이 너무 높으므로 좋은 염색체를 남길 확률 더 증가
    # 제일 좋은 것이 확률 1/2 -> [1/2, 1/4, 1/8, 1/16 ...]
    pro = [1.5 ** i for i in range(POP_SIZE - 1, -1, -1)]
    pick = rd.uniform(0, sum(pro))
    current = 0
    for c in range(len(spots)):
        current += pro[c]
        # 룰렛 알고리즘
        if current > pick:
            return pop[c]
    return pop[-1]


def crossover(pop): # 교배
    father = select(pop) # 룰렛 알고리즘으로 두 염색체 선택
    mother = select(pop)
    start, end = sorted(rd.sample(range(len(spots) + 1), 2)) # 0 ~ SIZE 중 무작위로 2개 인덱스 선택

    child = [None] * len(spots)
    for i in range(start, end): # 3부분으로 나눠 중간은 father, 양 끝은 mother 염색체 물려받음
        child[i] = father.genes[i]

    for i in range(len(spots)):
        for j in range(len(spots)):
            if mother.genes[j] not in child:
                child[i] = mother.genes[j]
    return child


def mutate(c): # 돌연변이
    if rd.random() < MUT_RATE:
        if rd.random() < 0.7: # 상호 교환 연산자 - 단순히 두 도시 변경
            a, b = rd.sample(range(len(spots)), 2)
            c.genes[a], c.genes[b] = c.genes[b], c.genes[a]
        else: # 역치 연산자 - 두 점을 선택 후 그 사이의 순서 변경
            a, b = sorted(rd.sample(range(len(spots) + 1), 2))
            if a != 0:
                c.genes = c.genes[:a] + c.genes[b-1:a-1:-1] + c.genes[b:]
            else:
                c.genes = c.genes[b-1::-1] + c.genes[b:]

def getProgress(fitnessMean, fitnessBest, population): # 추이 반영 - 알고리즘과는 상관없음
    fitnessSum = 0 # 평균을 구하기 위한 합계
    for c in population:
        fitnessSum += c.getFitness()
    fitnessMean.append(fitnessSum / POP_SIZE) # 세대의 평균 적합도 추이
    fitnessBest.append(population[0].getFitness()) # 세대의 적합도가 가장 좋은 염색체의 적합도 추이

def initialDistance(): # 각 장소의 거리들 초기화
    global distance
    distance = [[None] * len(spots) for _ in range(len(spots))]
    for i in range(len(spots)):
        for j in range(len(spots)):
            distance[i][j] = spots[i].distanceTo(spots[j])


def main(): # 메인함수
    population = [] # 염색체들
    fitnessMean = [] # 평균 적합도 추이
    fitnessBest = [] # 첫번째 염색체의 적합도 추이
    bestGene = None # 최단거리를 담고있는 염색체
    worstGene = None # 최장거리를 담고있는 염색체

    map_original = map.chooseSpot(spots)
    initialDistance()

    for _ in range(POP_SIZE): # 염색체들 초기화
        population.append(Chromosome())
    population.sort(key = lambda x: x.getFitness())
    bestGene = population[0]
    worstGene = population[-1]

    generation = 0
    while 1:
        print("세대 번호 =", generation)
        print_p(population)
        generation += 1
        
        if generation == MAX_GENE: # limit까지 도달할 경우
            getProgress(fitnessMean, fitnessBest, population) # 차트를 위해 평균 적합도, 최적 적합도 추이 반영
            break
        
        new_pop = []
        for _ in range(POP_SIZE): # 교배를 하여 새로운 염색체 생성
            child = crossover(population)
            new_pop.append(Chromosome(child))
        population = new_pop.copy() # 기존 집합을 새로운 염색체 집합으로 교체

        for c in population: # 확률적으로 돌연변이 연산 수행
            mutate(c)

        population.sort(key = lambda x: x.cal_fitness()) # 적합도에 따라 정렬

        if bestGene.getFitness() > population[0].getFitness(): # 최단거리 최장거리 갱신
            bestGene = population[0]
        if worstGene.getFitness() < population[-1].getFitness():
            worstGene = population[-1]

        if not map.updateUI(map_original, generation, bestGene, spots): # UI 업데이트
            break
        
        getProgress(fitnessMean, fitnessBest, population) # 차트를 위해 평균 적합도, 최적 적합도 추이 반영
        
    chart.drawChart(fitnessMean, fitnessBest, generation, bestGene, worstGene) # 마지막으로 차트 그리기
    # main함수 끝

if __name__ == '__main__':
    main()