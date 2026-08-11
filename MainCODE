import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dynamixel_sdk import *

# ==========================================
# 1. U2D2 및 모터 설정
# ==========================================
DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

ADDR_OPERATING_MODE = 11        # 운영 모드 (3: 위치 제어)
ADDR_PROFILE_ACCELERATION = 108 # 모터 가속도
ADDR_PROFILE_VELOCITY = 112     # 모터 최고 속도
ADDR_TORQUE_ENABLE = 64         # 토크 스위치
ADDR_GOAL_POSITION = 116        # 목표 위치 (4 bytes)
LEN_GOAL_POSITION = 4

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
POSITION_CONTROL_MODE = 3
DEFAULT_POS = 2048              # 기본 위치 (center)

# 속도/가속도 상향 (큰 동작을 빠르게 수행하기 위함)
SMOOTH_VELOCITY = 350           
SMOOTH_ACCEL = 120              

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

if not portHandler.openPort() or not portHandler.setBaudRate(BAUDRATE):
    print("❌ U2D2 연결 실패!")
    exit()

print("⚙️ 전체 모터(ID 1~19) 초기화 및 토크 인가 중...")

for dxl_id in range(1, 20):
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    time.sleep(0.001)
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, POSITION_CONTROL_MODE)
    time.sleep(0.001)
    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PROFILE_ACCELERATION, SMOOTH_ACCEL)
    time.sleep(0.001)
    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PROFILE_VELOCITY, SMOOTH_VELOCITY)
    time.sleep(0.001)
    
    comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    if comm_result != COMM_SUCCESS:
        print(f"⚠️ ID {dxl_id} 모터 토크 인가 실패 (통신 에러)")
    time.sleep(0.001)

print("✅ 모터 설정 완료!")

prev_positions = {dxl_id: DEFAULT_POS for dxl_id in range(1, 20)}

# ==========================================
# 2. 각도 계산 및 매핑 함수
# ==========================================
def calculate_angle(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def angle_to_dxl_pos(angle, min_deg=10, max_deg=170, gain=2.0, invert_dir=False):
    angle = max(min_deg, min(max_deg, angle))
    norm = (max_deg - angle) / (max_deg - min_deg)
    norm = min(1.0, norm * gain)
    
    if invert_dir:
        target_pos = int(2048 - norm * (2048 - 1024))
    else:
        target_pos = int(2048 + norm * (3072 - 2048))
        
    return target_pos

# ==========================================
# 3. MediaPipe Tasks 및 카메라 설정
# ==========================================
base_options = python.BaseOptions(model_asset_path='/home/mirae/robot_hand/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20),
    (5,9), (9,13), (13,17)
]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

timestamp = 0
prev_time = time.time()

# ==========================================
# 4. 메인 제어 루프
# ==========================================
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ 카메라 프레임 읽기 실패!")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp += 1
        result = landmarker.detect_for_video(mp_image, timestamp)

        target_positions = {dxl_id: DEFAULT_POS for dxl_id in range(1, 20)}
        hand_detected = False

        if result.hand_landmarks:
            hand_detected = True
            lm = result.hand_landmarks[0]

            # ----------------------------------------------------
            # 1. 엄지 (ID 1, 2, 3) - gain을 파격적으로 올려 크게 동작
            # ----------------------------------------------------
            ang_t1 = calculate_angle(lm[2], lm[3], lm[4])
            ang_t2 = calculate_angle(lm[1], lm[2], lm[3])
            ang_t3 = calculate_angle(lm[0], lm[1], lm[2])

            target_positions[1] = angle_to_dxl_pos(ang_t1, min_deg=5, max_deg=170, gain=5.0, invert_dir=True)
            target_positions[2] = angle_to_dxl_pos(ang_t2, min_deg=5, max_deg=170, gain=4.5, invert_dir=True)
            target_positions[3] = angle_to_dxl_pos(ang_t3, min_deg=5, max_deg=170, gain=4.0, invert_dir=True)

            # ----------------------------------------------------
            # 2. 검지 (ID 4, 5, 6 / ID 7 고정) - 더 많이 꺾이도록 gain 상향
            # ----------------------------------------------------
            ang_i4 = calculate_angle(lm[0], lm[5], lm[6])  
            ang_i5 = calculate_angle(lm[5], lm[6], lm[7])  
            ang_i6 = calculate_angle(lm[6], lm[7], lm[8])  

            target_positions[4] = angle_to_dxl_pos(ang_i4, min_deg=10, max_deg=170, gain=3.0)
            target_positions[5] = angle_to_dxl_pos(ang_i5, min_deg=10, max_deg=170, gain=3.2)
            target_positions[6] = angle_to_dxl_pos(ang_i6, min_deg=5,  max_deg=170, gain=4.5)
            target_positions[7] = 2048  # 고정

            # ----------------------------------------------------
            # 3. 중지 (ID 8, 9, 10 / ID 11 고정)
            # ----------------------------------------------------
            target_positions[8]  = angle_to_dxl_pos(calculate_angle(lm[0], lm[9], lm[10]), min_deg=15, gain=2.5)
            target_positions[9]  = angle_to_dxl_pos(calculate_angle(lm[9], lm[10], lm[11]), min_deg=15, gain=2.5)
            target_positions[10] = angle_to_dxl_pos(calculate_angle(lm[10], lm[11], lm[12]), min_deg=10, gain=3.5)
            target_positions[11] = 2048  # 고정

            # ----------------------------------------------------
            # 4. 약지 (ID 12, 13, 14 / ID 15 고정)
            # ----------------------------------------------------
            target_positions[12] = angle_to_dxl_pos(calculate_angle(lm[0], lm[13], lm[14]), min_deg=15, gain=2.5)
            target_positions[13] = angle_to_dxl_pos(calculate_angle(lm[13], lm[14], lm[15]), min_deg=15, gain=2.5)
            target_positions[14] = angle_to_dxl_pos(calculate_angle(lm[14], lm[15], lm[16]), min_deg=10, gain=3.5)
            target_positions[15] = 2048  # 고정

            # ----------------------------------------------------
            # 5. 새끼 (ID 16, 17, 18 / ID 19 고정)
            # ----------------------------------------------------
            target_positions[16] = angle_to_dxl_pos(calculate_angle(lm[0], lm[17], lm[18]), min_deg=15, gain=2.5)
            target_positions[17] = angle_to_dxl_pos(calculate_angle(lm[17], lm[18], lm[19]), min_deg=15, gain=2.5)
            target_positions[18] = angle_to_dxl_pos(calculate_angle(lm[18], lm[19], lm[20]), min_deg=10, gain=3.5)
            target_positions[19] = 2048  # 고정

            # ----------------------------------------------------
            # UI 시각화
            # ----------------------------------------------------
            h, w, _ = frame.shape
            pts = [(int(pt.x * w), int(pt.y * h)) for pt in lm]

            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, pts[start_idx], pts[end_idx], (255, 180, 0), 2, cv2.LINE_AA)

            for idx, pt in enumerate(pts):
                color = (0, 255, 255) if idx in [4, 8, 12, 16, 20] else (0, 255, 0)
                cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)

        # ----------------------------------------------------
        # 동적 필터링 (Alpha 조정)
        # ----------------------------------------------------
        final_positions = {}
        for dxl_id in range(1, 20):
            # 엄지(1~3번)는 커진 움직임을 즉각 따라가도록 alpha를 0.55로 상향
            # 검지(4~6번)는 0.40, 그 외는 0.30 적용
            if dxl_id in [1, 2, 3]:
                alpha = 0.55
            elif dxl_id in [4, 5, 6]:
                alpha = 0.40
            else:
                alpha = 0.30
            
            smooth_pos = int(prev_positions[dxl_id] * (1.0 - alpha) + target_positions[dxl_id] * alpha)
            final_positions[dxl_id] = smooth_pos
            prev_positions[dxl_id] = smooth_pos

        # SyncWrite 명령 전송
        groupSyncWrite.clearParam()
        for dxl_id, pos in final_positions.items():
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(pos)),
                DXL_HIBYTE(DXL_LOWORD(pos)),
                DXL_LOBYTE(DXL_HIWORD(pos)),
                DXL_HIBYTE(DXL_HIWORD(pos))
            ]
            groupSyncWrite.addParam(dxl_id, param_goal_position)

        groupSyncWrite.txPacket()

        # ----------------------------------------------------
        # UI 대시보드
        # ----------------------------------------------------
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 55), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, "ROBOT HAND SYSTEM v2.0", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        if hand_detected:
            cv2.rectangle(frame, (frame.shape[1] - 220, 10), (frame.shape[1] - 15, 45), (0, 180, 80), -1)
            cv2.putText(frame, "HAND DETECTED", (frame.shape[1] - 205, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (frame.shape[1] - 220, 10), (frame.shape[1] - 15, 45), (40, 40, 180), -1)
            cv2.putText(frame, "SEARCHING...", (frame.shape[1] - 200, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Robot Hand Controller', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\n종료 처리 중...")
    groupSyncWrite.clearParam()
    for dxl_id in range(1, 20):
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    portHandler.closePort()
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 안전하게 토크 OFF 및 종료 완료")
