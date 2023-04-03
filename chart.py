import matplotlib.pyplot as plt

def drawChart(fitnessMean, fitnessBest, generation, bestGene, worstGene): # 추이 그래프 그리기 - 알고리즘과는 상관없음
    plt.plot(range(len(fitnessMean)), fitnessMean, "-b", label="mean fitness", linewidth="0.5")
    plt.plot(range(len(fitnessMean)), fitnessBest, "-r", label="best fitness", linewidth = "0.5")
    plt.axis([0, generation * 1.1, bestGene.getFitness() * 0.9, worstGene.getFitness()])
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.legend(loc='upper left')
    plt.show()
    