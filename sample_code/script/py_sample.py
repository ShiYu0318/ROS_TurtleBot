#!/usr/bin/env python3
# coding=utf-8
"""
py_sample：光達避障範例節點

功能：
  1. 訂閱 /scan（LaserScan），輸出機器人正前方（按 w 前進的方向）±15 度的光達資料
  2. 節點啟動後，透過 /cmd_vel（TwistStamped）讓機器人以低速持續往前走
  3. 前方 ±15 度內最近的障礙物距離小於 STOP_DISTANCE 時停車，避免撞上

資料流：
  /scan ──> scan_callback()    更新 self.front_min（前方最近距離）
  Timer ──> control_callback() 依 self.front_min 決定速度 ──> /cmd_vel
"""
import math

import rclpy
from geometry_msgs.msg import TwistStamped  # 速度指令；teleop_key.py 也是發這個型別
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # 感測器用的 QoS（Best Effort）
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan  # 光達掃描資料

# ---------------- 可調整的常數 ----------------
# 除了 FRONT_HALF_WIDTH_DEG，其他都可以在執行時用 ROS 參數覆蓋（見 __init__）
FRONT_HALF_WIDTH_DEG = 15.0  # 正前方 ±15 度
LINEAR_SPEED = 0.05          # m/s，慢速前進（teleop 上限：minibot 0.28、turtlebot 0.22）
STOP_DISTANCE = 0.30         # m，前方最近距離小於此值就停車（從光達中心量起，不是車頭）
RESUME_DISTANCE = 0.40       # m，障礙物離開到此距離外才繼續走，避免在門檻附近走走停停
SCAN_TIMEOUT = 0.5           # s，超過這麼久沒收到 /scan 就停車（光達斷線時的保護）


def RAD2DEG(r):
    """弧度轉角度。"""
    return r * 180 / math.pi


def front_points(scan, half_width_deg, front_offset_deg=0.0):
    """回傳正前方 ±half_width_deg 內的 (角度[deg], 距離[m])，由右(-)到左(+)排序。

    每個點的角度用 angle_min + i * angle_increment 計算，所以 minibot（每 0.5 度一點）
    和 turtlebot（每 1 度一點）都適用，也不受 angle_min 從 0 或 -pi 開始的影響。

    參數：
      scan             LaserScan 訊息
      half_width_deg   要取的半角，15 代表 -15 ~ +15 度
      front_offset_deg 光達 0 度與車頭方向的夾角；光達反裝時設 180
    """
    points = []
    for i, r in enumerate(scan.ranges):
        # 第 i 個點在光達座標系的角度（弧度），再扣掉安裝偏移，變成相對車頭的角度
        angle = scan.angle_min + i * scan.angle_increment - math.radians(front_offset_deg)
        # math.remainder 把角度折回 [-pi, pi]，例如 350 度會變成 -10 度，
        # 這樣車頭右側（-15 ~ 0 度）和左側（0 ~ 15 度）才能用同一個條件判斷
        degree = RAD2DEG(math.remainder(angle, 2 * math.pi))
        # 加 1e-3 的容許誤差：浮點數累加後 15 度可能變成 15.0000001 而被漏掉
        if abs(degree) <= half_width_deg + 1e-3:
            points.append((degree, r))
    points.sort()  # 依角度排序，輸出時由右到左
    return points


def min_distance(scan, points):
    """前方最近的有效距離；inf（沒打到東西）視為很遠，nan 和小於 range_min 的讀值（如 0）忽略。"""
    # 光達量不到時，不同機型會回傳 inf、nan 或 0：
    #   inf ：前方空曠，保留下來，min() 時自然會被其他較近的點取代
    #   nan / 0 / 小於 range_min：無效讀值，直接丟掉，否則會被誤判成「障礙物就在眼前」
    valid = [r for _, r in points if not math.isnan(r) and r >= scan.range_min]
    # 全部無效或全部是 inf 時回傳 inf，代表前方沒有障礙物
    return min(valid, default=math.inf)


class PySample(Node):
    def __init__(self):
        super().__init__('py_sample')  # 節點名稱，ros2 node list 會看到 /py_sample

        # ---- ROS 參數：不用改程式就能調整，例如 ----
        #   ros2 run sample_code py_sample.py --ros-args -p linear_speed:=0.03 -p stop_distance:=0.35
        # 注意：數值要寫成小數（0.0 而不是 0），否則型別不符會報錯
        # 若光達不是正對車頭（按 w 的方向）安裝，可用 --ros-args -p front_offset_deg:=180.0 修正
        self.front_offset_deg = self.declare_parameter('front_offset_deg', 0.0).value
        self.linear_speed = self.declare_parameter('linear_speed', LINEAR_SPEED).value
        self.stop_distance = self.declare_parameter('stop_distance', STOP_DISTANCE).value
        self.resume_distance = self.declare_parameter('resume_distance', RESUME_DISTANCE).value

        # ---- 節點狀態 ----
        self.front_min = math.inf    # 最近一次掃描中，前方 ±15 度內的最近距離（m）
        self.last_scan_time = None   # 最近一次收到 /scan 的時間；None 代表還沒收到過
        self.blocked = False         # 目前是否因為障礙物而停車
        self.state = None            # 上一次的狀態代碼，用來判斷狀態是否改變

        # 建立 Publisher，發布速度指令到 /cmd_vel（queue size 10，預設 Reliable）
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # 建立 Timer，0.1秒，負責送出速度指令
        # 底盤通常要求持續收到指令，所以即使速度不變也要固定頻率一直發
        self.create_timer(0.1, self.control_callback)

        # 建立 Subscriber，以 Best Effort 訂閱 /scan
        # 光達驅動是用 Best Effort 發布，這裡若用 Reliable 會因 QoS 不相容而收不到資料
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)

    def scan_callback(self, scan):
        """每收到一筆光達資料就執行：取出前方 ±15 度、更新最近距離、印出結果。"""
        points = front_points(scan, FRONT_HALF_WIDTH_DEG, self.front_offset_deg)
        self.front_min = min_distance(scan, points)
        self.last_scan_time = self.get_clock().now()  # 記錄時間，給 control_callback 判斷資料是否過期

        # 每個點一行：角度（右負左正）、距離；最後一行是摘要
        # 先組成一個字串再一次 print，比逐行 print 快，也避免和 log 交錯
        lines = [f'[LIDAR INFO]:angle-distance:[{degree:>5.1f}, {r:>5.3f}]' for degree, r in points]
        lines.append(f'[LIDAR INFO]:front ±{FRONT_HALF_WIDTH_DEG:.0f} deg, {len(points)} points, '
                     f'min distance {self.front_min:.3f} m')
        # flush=True：用 ros2 launch 執行時 stdout 可能被緩衝，強制立即輸出
        print('\n'.join(lines), flush=True)

    def control_callback(self):
        """每 0.1 秒執行：依前方距離決定前進或停車，並發布速度指令。"""
        # 情況 1：還沒收到過 /scan，或資料已經過期 → 看不到前方，安全起見停車
        # （Time 相減得到 Duration，用 nanoseconds 和秒數 * 1e9 比較）
        if self.last_scan_time is None or \
                (self.get_clock().now() - self.last_scan_time).nanoseconds > SCAN_TIMEOUT * 1e9:
            state, message = 'no_scan', 'no recent /scan, stopped'
            speed = 0.0
        else:
            # 情況 2：有最新的光達資料 → 用兩個門檻（遲滯）決定要不要停
            #   行進中：距離 < stop_distance   才停下
            #   停車中：距離 > resume_distance 才重新出發
            # 兩個門檻中間（0.30 ~ 0.40 m）維持原狀態，避免距離在 0.30 m 附近
            # 跳動時，機器人一直走走停停
            if self.blocked and self.front_min > self.resume_distance:
                self.blocked = False
            elif not self.blocked and self.front_min < self.stop_distance:
                self.blocked = True

            if self.blocked:
                state, message = 'blocked', f'obstacle at {self.front_min:.3f} m, stopped'
                speed = 0.0
            else:
                state, message = 'moving', f'path clear, moving forward at {self.linear_speed:.2f} m/s'
                speed = self.linear_speed

        self.publish_speed(speed)

        # 只在狀態改變時印 log，避免洗版
        # 用狀態代碼比較，而不是 message：message 裡的距離每次都不同
        if state != self.state:
            self.get_logger().info(message)
        self.state = state

    def publish_speed(self, speed):
        """發布直線速度（m/s）到 /cmd_vel；只往前直走，所以角速度固定為 0。"""
        twist_stamped = TwistStamped()  # 新建訊息，所有欄位預設為 0
        twist_stamped.header.stamp = self.get_clock().now().to_msg()  # 時間戳記
        twist_stamped.twist.linear.x = float(speed)  # x 軸是車頭方向，正值往前（等同按 w）
        self.cmd_pub.publish(twist_stamped)


def main(args=None):
    # 關掉 rclpy 內建的 Ctrl+C 處理，讓 finally 裡的 context 還有效，才能送出停車指令
    # （預設情況下 Ctrl+C 會先關掉 context，之後 publish 就會失敗，機器人會照最後的速度繼續走）
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)

    node = PySample()

    try:
        rclpy.spin(node)  # 持續處理 Timer 和 Subscriber 的 callback，直到被中斷
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    finally:
        node.publish_speed(0.0)  # 結束前讓機器人停下
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
