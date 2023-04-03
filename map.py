import cv2
import tsp_genetic

def mouse_callback(event, x, y, flags, param): # 마우스 콜백함수
    if event == cv2.EVENT_LBUTTONDOWN: # 마우스로 클릭된 곳에 빨간 원을 그림
        print(x, y)
        cv2.circle(param[0], center=(x, y), radius=10, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.imshow('soongsil map', param[0])

        param[1].append(tsp_genetic.Spot(x, y))

def chooseSpot(spots): # 마우스로 장소를 입력받는 함수
    map_original = cv2.imread('soongsil_skyview.png')
    cv2.imshow('soongsil map', map_original)
    cv2.setMouseCallback('soongsil map', mouse_callback, [map_original, spots])
    cv2.waitKey(0)
    return map_original

def updateUI(map_original, generation, bestGene, spots): # UI 갱신
    map_result = map_original.copy()

    for i in range(1, tsp_genetic.Spot.num): # 장소들을 잇는 선 그리기
        cv2.line(map_result,
            pt1=(spots[bestGene.genes[i-1]].x, spots[bestGene.genes[i-1]].y),
            pt2=(spots[bestGene.genes[i]].x, spots[bestGene.genes[i]].y),
            color=(255, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA
        )
    # 왼쪽 위 텍스트 갱신
    cv2.putText(map_result, org=(10, 25), text='Generation: %d' % generation, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(map_result, org=(10, 50), text='Distance: %.2fm' % bestGene.getFitness(), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(map_result, org=(10, 75), text='Press q to exit', fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('map', map_result)

    if cv2.waitKey(1) == ord('q'): # 대기시간
        return False
    return True