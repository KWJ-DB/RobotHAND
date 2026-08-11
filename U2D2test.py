/home/mirae/robot_hand/env/bin/python -c "
from dynamixel_sdk import *
p = PortHandler('/dev/ttyUSB0')
pkt = PacketHandler(2.0)
if p.openPort() and p.setBaudRate(57600):
    volt, _, _ = pkt.read2ByteTxRx(p, 1, 144) # Present Voltage (ADDR 144)
    err, _, _ = pkt.read1ByteTxRx(p, 1, 70)    # Hardware Error Status (ADDR 70)
    print(f'현재 입력 전압: {volt/10.0} V')
    print(f'하드웨어 에러 코드: {err}')
    p.closePort()
"
