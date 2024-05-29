# HAND MOTION DRAWING 
+ Language : Python3.7
+ Framework & Model : Mediapipe-hands, MLP, Tensorflow-Keras
 ---
 ## Program Architecture
![architecture](./progtam_architecture.png)
---
## ML Model
 ### Dataset
+ hands point to degree
 ```
 for res in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] 
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] 
            v = v2 - v1 # [20,3]

            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

 ```
1. 포인트간 벡터의 차를 이용해 마디별 백터 추출
2. 백터의 정규화
3. 정규화된 값들의 내적을 통해 마디 사이 각도 추출
---
### Model Input Separate
two-hands motion
```
def twohand_mode(landmark1, landmark2):
    mode = False

    st1 = prc.get_distance(prc.point_to_list(landmark1[0]), prc.point_to_list(landmark1[9]))
    st2 = prc.get_distance(prc.point_to_list(landmark2[0]), prc.point_to_list(landmark2[9]))  
    standard = ((st1+st2) / 2 ) * 3

    m_pt1 = [(landmark1[0].x + landmark1[9].x)/2,
                (landmark1[0].y + landmark1[9].y)/2,
                (landmark1[0].z + landmark1[9].z)/2]
    m_pt2 = [(landmark2[0].x + landmark2[9].x)/2,
                (landmark2[0].y + landmark2[9].y)/2,
                (landmark2[0].z + landmark2[9].z)/2]
    distance = prc.get_distance(m_pt1, m_pt2)

    if distance <= standard:
        mode = True

    return distance, mode
```
+ 각 손의 0,9포인트 거리가 기준 거리보다 작으면 two-hands motion으로 사전 분류 
+ 기준 거리: 손바닥과 중지의 길이가 비슷하다는 가정 -> 기준점(손바닥 중점)들 사이의 거리가 각 손의 0(손꿈치),9(중지시작)포인트 거리의 3배 이내.

one-hand motion
```
temp_list = []
for i in self.about_twohand_list:
    if (i[1] not in temp_list) and (i[2] not in temp_list):
        self.model_list.append( [i[1], i[2]] )
        temp_list.append(i[1])
        temp_list.append(i[2])      

if len(temp_list)!=self.num+1:
    for i in range(self.num+1):
        if i not in temp_list:
            self.model_list.append([i])
```
+ two-hands 동작 분류 후 남은 동작은 one-hand 동작으로 분류.
---
### Model
one-hand Model
```
model = keras.models.Sequential()
model.add(Dense(32, input_shape=(15,), activation='relu'))
model.add(Dense(6, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])  

```
+ input : 관절각 데이터 크기(15.)
+ activation function : softmax
+ loss function : Categorical Crossentropy


two-hand Model
```
model = keras.models.Sequential()
model.add(Dense(32, input_shape=(30,), activation='relu'))
model.add(Dense(9, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
+ input : 두 손 관절각 데이터 크기(32.)
+ activation function : softmax
+ loss function : Categorical Crossentropy
---
## Display Output

### Masking Background and PaintImage
```
def s_masking (img, x, y, bg, rows, cols):
  
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  ret,img_mask = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)
  img_mask_inv = cv2.bitwise_not(img_mask)
  img_roi = bg[y:y+rows, x:x+cols]
  img1 = cv2.bitwise_and(img, img, mask = img_mask_inv) 
  img2 = cv2.bitwise_and(img_roi, img_roi, mask=img_mask)
  dst = cv2.add(img1, img2)
  
  return dst
```
+ 배경과 출력 그림(이미지) 합성
---
### Get Path for Dynamic Image
standard path
```
# one of root making
def goksun1 (xc,yc):
  r = []
  ban = 100
  yr = 0
  way = 0
  for i in range(0,ban):
    xr = xc+i
    yr = round(0.004*(xr-xc)**2)+yc
    r.append([xr, yr])
    if xr < xc:
      way = -1
    elif xr > xc:
      way = 1
```
+ 경로 생성 : 기울기 0.004를 취하는 2차함수그래프의 자취
+ 이동폭 : x축 기준 100의 양만큼 증가 

wall path
```
# one of root making for wall
def goksun1_1 (xc,yc):
  r = []
  ban = 50
  yr = 0
  way = 0
  for i in range(0,ban):
    xr = xc+i
    yr = yc
    r.append([xr, yr])
    if xr < xc:
      way = -1
    elif xr > xc:
      way = 1
  return r, ban, way
```
+ 이미지가 디스플레이 말단 도착 시 수직 혹은 수평방향으로 50의 양만큼 이동  
