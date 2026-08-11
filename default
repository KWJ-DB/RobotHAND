import time
from dynamixel_sdk import *

# ==========================================
# U2D2 통신 및 모터 레지스터 주소 설정
# ==========================================
DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

ADDR_OPERATING_MODE = 11      # 운영 모드 (3: 위치 제어)
ADDR_TORQUE_ENABLE = 64       # 토크 스위치
ADDR_GOAL_POSITION = 116      # 목표 위치 (4 bytes)
LEN_GOAL_POSITION = 4

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
POSITION_CONTROL_MODE = 3
DEFAULT_POSITION = 2048       # Center Position (180도 기본값)

# 포트 및 패킷 핸들러 초기화
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

if not portHandler.openPort():
    print(f"❌ {DEVICENAME} 포트를 열 수 없습니다. 연결을 확인하세요.")
    exit()

if not portHandler.setBaudRate(BAUDRATE):
    print(f"❌ 보드레이트를 {BAUDRATE}로 설정하지 못했습니다.")
    portHandler.closePort()
    exit()

print("🔄 모터 초기화 및 기본값(2048) 복귀 시작...")

try:
    # 1. 19개 모터 위치 제어 모드 설정 및 토크 ON
    for dxl_id in range(1, 20):
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, POSITION_CONTROL_MODE)
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

    # 2. SyncWrite 패킷 생성 (모든 모터 ID 1~19에 2048 전달)
    groupSyncWrite.clearParam()
    param_goal_position = [
        DXL_LOBYTE(DXL_LOWORD(DEFAULT_POSITION)),
        DXL_HIBYTE(DXL_LOWORD(DEFAULT_POSITION)),
        DXL_LOBYTE(DXL_HIWORD(DEFAULT_POSITION)),
        DXL_HIBYTE(DXL_HIWORD(DEFAULT_POSITION))
    ]

    for dxl_id in range(1, 20):
        groupSyncWrite.addParam(dxl_id, param_goal_position)

    # 3. 목표 위치 전송
    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"⚠️ 통신 에러: {packetHandler.getTxRxResult(dxl_comm_result)}")
    else:
        print("✅ 모든 모터에 기본 위치(2048) 전송 완료!")

    # 모터가 이동할 시간 대기 (1.5초)
    time.sleep(1.5)

finally:
    # 4. 모터 안전 종료 (토크 OFF)
    print("🔒 안전을 위해 모터 토크(Torque)를 해제합니다...")
    for dxl_id in range(1, 20):
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    
    groupSyncWrite.clearParam()
    portHandler.closePort()
    print("🎉 손 리셋 작업이 완료되었습니다.")
