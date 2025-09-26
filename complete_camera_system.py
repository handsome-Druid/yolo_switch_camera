# coding=utf-8
"""
完整版智能摄像头切换系统
- USB摄像头检测粉色喷水壶位置
- 自动切换到对应的网络相机
- 支持手动切换
"""
import cv2
import numpy as np
import time
import threading
import sys
import os
import json

# 添加本地ultralytics路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ultralytics'))
from ultralytics import YOLO

# 添加相机模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'camera'))

class SmartCameraSystem:
    def run(self):
        """运行智能摄像头系统"""
        print("=" * 50)
        print("🎯 智能摄像头切换系统")
        print("=" * 50)
        print("功能说明:")
        print("• USB摄像头单独窗口持续采集+检测 (基准视角)")
        print("• 只根据粉色喷水壶左右位置切换网络摄像头 0 / 1")
        print("• 中间区域不触发切换 (保持当前网络画面)")
        print("• 手动切换网络相机: 1=network_0  2=network_1")
        print("=" * 50)
        # 初始化相机
        print("🔧 正在初始化相机...")
        usb_ok = self.initialize_usb_camera()
        network_ok = self.initialize_network_cameras()
        if not usb_ok:
            print("❌ USB摄像头不可用，程序退出")
            return
        if not network_ok:
            print("⚠️  网络相机不可用，只能显示USB摄像头")
        print(f"✅ 系统初始化完成")
        print(f"   - USB摄像头: {'✓' if usb_ok else '✗'}")
        print(f"   - 网络相机: {len(self.network_cameras)}个")
        print()
        self.running = True
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.display_thread = threading.Thread(target=self.display_loop)
        self.detection_thread.daemon = True
        self.display_thread.daemon = True
        try:
            self.detection_thread.start()
            self.display_thread.start()
            print("🚀 系统运行中... (按Q退出)")
            self.display_thread.join()
        except KeyboardInterrupt:
            print("\n🛑 收到键盘中断信号")
        finally:
            self.running = False
            self.cleanup()
    def __init__(self):
        """初始化智能摄像头系统"""
        print("🚀 初始化智能摄像头切换系统...")
        
        # 加载YOLO模型
        try:
            self.model = YOLO('yolo12n.pt')
            print("✓ YOLO模型加载成功")
        except Exception as e:
            print(f"❌ YOLO模型加载失败: {e}")
            return
        
        # 相机相关
        self.usb_camera = None
        self.network_cameras = []
        # 当前网络显示相机 (network_0 / network_1)
        self.current_display = "network_0"
        
        # 检测相关参数
        self.target_classes = ['bottle', 'vase', 'cup', 'bowl']  # 可能的喷水壶类别
        self.detection_confidence = 0.4
        self.position_threshold = 0.35  # 左右分界线位置（不再使用 center 回退）
        
        # 检测稳定性控制
        self.detection_stable_frames = 3  # 需要连续几帧检测到才认为稳定
        self.no_detection_frames = 5      # 需要连续几帧检测不到才认为丢失
        self.current_stable_frames = 0
        self.current_no_detect_frames = 0
        self.stable_detection_state = False  # 当前稳定的检测状态
        
        # 状态控制
        self.running = False
        self.switch_cooldown = 1.5  # 切换冷却时间（秒）
        self.last_switch_time = 0
        self.detection_stable_time = 0.8  # 检测稳定时间
        self.last_detection_time = 0
        self.last_position = None
        
        # 线程控制
        self.detection_thread = None
        self.display_thread = None
        self.lock = threading.Lock()
        
        # 统计信息
        self.detection_count = 0
        self.switch_count = 0
        # 网络相机显示/色彩相关配置
        # flip_mode: none / vertical / horizontal / both
        self.network_flip_mode = 'vertical'  # 默认上下翻转修正倒置
        # 自动白平衡默认开启，如果SDK不支持则回退到简单软件白平衡
        self.enable_auto_wb = True
        self.software_wb_backup = True  # 当硬件白平衡不可用或关闭时使用软件WB
        # 最近一次键盘操作提示时间
        self._last_help_print = 0
        # Pillow 中文绘制支持
        self._pillow_available = False
        self._font_cache = {}
        try:
            from PIL import Image, ImageDraw, ImageFont  # noqa: F401
            self._pillow_available = True
        except Exception:
            print("⚠️ 未安装 Pillow，界面中文将显示为 ??? ，可执行: pip install pillow")
        # 记录最后一次检测状态供主窗口显示
        self.last_detect_found = False
        self.last_detect_position = None
        # USB 画中画缓存与开关
        self.last_usb_display = None
        self.show_usb_pip = True
        # 分相机 zoom/sharpen 参数
        self.camera_params = {
            'network_0': {'zoom': 0, 'sharpen': 0},
            'network_1': {'zoom': 0, 'sharpen': 0}
        }
        # 配置文件路径
        self.config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        self.load_config()

    def _get_font(self, size=22):
        if not self._pillow_available:
            return None
        key = size
        if key in self._font_cache:
            return self._font_cache[key]
        from PIL import ImageFont
        candidates = [
            r"c:/Windows/Fonts/msyh.ttc",
            r"c:/Windows/Fonts/msyh.ttf",
            r"c:/Windows/Fonts/simhei.ttf",
            r"c:/Windows/Fonts/simsun.ttc",
            r"c:/Windows/Fonts/msyhbd.ttc",
        ]
        font = None
        for path in candidates:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, size=size)
                    break
                except Exception:
                    pass
        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
        self._font_cache[key] = font
        return font

    def draw_text_cn(self, img, text, xy, size=22, color=(0, 255, 255)):
        if text is None:
            return
        if not self._pillow_available:
            cv2.putText(img, text.encode('utf-8', 'ignore').decode('ascii', 'ignore') or '?', xy, cv2.FONT_HERSHEY_SIMPLEX, size/32.0, color, 2, cv2.LINE_AA)
            return
        from PIL import Image, ImageDraw
        font = self._get_font(size)
        if font is None:
            cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size/32.0, color, 2, cv2.LINE_AA)
            return
        # OpenCV(BGR) -> PIL(RGB)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
        # PIL 回到 OpenCV
        img[:, :, :] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    # ---------------- 配置管理 ----------------
    def load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                self.network_flip_mode = cfg.get('network_flip_mode', self.network_flip_mode)
                self.enable_auto_wb = cfg.get('enable_auto_wb', self.enable_auto_wb)
                self.show_usb_pip = cfg.get('show_usb_pip', self.show_usb_pip)
                cam_cfg = cfg.get('camera_params')
                if cam_cfg:
                    for k in ['network_0', 'network_1']:
                        if k in cam_cfg:
                            self.camera_params[k]['zoom'] = int(cam_cfg[k].get('zoom', 0))
                            self.camera_params[k]['sharpen'] = int(cam_cfg[k].get('sharpen', 0))
                else:
                    # 兼容旧字段
                    z = int(cfg.get('zoom_percent', 0))
                    s = int(cfg.get('sharpen_level', 0))
                    self.camera_params['network_0']['zoom'] = z
                    self.camera_params['network_1']['zoom'] = z
                    self.camera_params['network_0']['sharpen'] = s
                    self.camera_params['network_1']['sharpen'] = s
                print(f"🗂️ 已加载配置: {cfg}")
        except Exception as e:
            print(f"⚠️ 加载配置失败: {e}")

    def save_config(self):
        data = {
            'network_flip_mode': self.network_flip_mode,
            'enable_auto_wb': self.enable_auto_wb,
            'show_usb_pip': self.show_usb_pip,
            'camera_params': self.camera_params
        }
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存配置失败: {e}")
        
    def initialize_usb_camera(self):
        """初始化USB摄像头"""
        try:
            self.usb_camera = cv2.VideoCapture(0)
            if not self.usb_camera.isOpened():
                print("❌ 无法打开USB摄像头")
                return False
            
            # 设置最佳参数
            self.usb_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.usb_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.usb_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.usb_camera.set(cv2.CAP_PROP_FPS, 30)
            
            # 获取实际分辨率
            width = int(self.usb_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.usb_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✓ USB摄像头初始化成功 ({width}x{height})")
            return True
            
        except Exception as e:
            print(f"❌ USB摄像头初始化失败: {e}")
            return False
    
    def reconnect_usb_camera(self):
        """重新连接USB摄像头"""
        try:
            # 关闭现有连接
            if self.usb_camera is not None:
                try:
                    self.usb_camera.release()
                except:
                    pass
                self.usb_camera = None
            
            # 等待设备稳定
            time.sleep(1.0)
            
            # 尝试重新初始化
            return self.initialize_usb_camera()
            
        except Exception as e:
            print(f"❌ USB摄像头重连失败: {e}")
            return False
    
    def initialize_network_cameras(self):
        """初始化网络相机"""
        try:
            import mvsdk
            
            # 枚举相机
            DevList = mvsdk.CameraEnumerateDevice()
            nDev = len(DevList)
            
            if nDev < 2:
                print(f"⚠️  只找到 {nDev} 个网络相机，建议至少2个")
                if nDev == 0:
                    return False
                    
            print(f"🔍 找到 {nDev} 个网络相机")
            
            # 初始化网络相机
            for i in range(min(2, nDev)):
                try:
                    camera_info = self.init_single_network_camera(DevList[i], i)
                    self.network_cameras.append(camera_info)
                    print(f"✓ 网络相机 {i} 初始化成功: {DevList[i].GetFriendlyName()}")
                except Exception as e:
                    print(f"❌ 网络相机 {i} 初始化失败: {e}")
                    
            return len(self.network_cameras) > 0
            
        except ImportError:
            print("❌ 无法导入mvsdk，网络相机功能不可用")
            return False
        except Exception as e:
            print(f"❌ 网络相机初始化失败: {e}")
            return False
    
    def init_single_network_camera(self, dev_info, camera_id):
        """初始化单个网络相机"""
        import mvsdk
        
        # 打开相机
        hCamera = mvsdk.CameraInit(dev_info, -1, -1)
        
        # 获取相机特性
        cap = mvsdk.CameraGetCapability(hCamera)
        
        # 判断是否为黑白相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
        
        # 设置输出格式
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
        
        # 设置连续采集模式
        mvsdk.CameraSetTriggerMode(hCamera, 0)
        
        # 设置曝光参数
        mvsdk.CameraSetAeState(hCamera, 0)
        mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)
        
        # 开始采集
        mvsdk.CameraPlay(hCamera)

        # 白平衡设置：尝试开启自动白平衡，若失败则标记不可用
        wb_supported = True
        try:
            mvsdk.CameraSetWbMode(hCamera, 1 if self.enable_auto_wb else 0)
        except Exception:
            wb_supported = False

        if wb_supported and self.enable_auto_wb:
            try:
                # 触发一次快速自动白平衡收敛
                mvsdk.CameraSetOnceWB(hCamera)
            except Exception:
                pass
        
        # 分配帧缓冲区
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
        pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
        
        return {
            'id': camera_id,
            'handle': hCamera,
            'capability': cap,
            'mono': monoCamera,
            'buffer': pFrameBuffer,
            'buffer_size': FrameBufferSize,
            'name': dev_info.GetFriendlyName(),
            'wb_supported': wb_supported,
            'awb_on': self.enable_auto_wb and wb_supported
        }
    
    def get_network_camera_frame(self, camera_info):
        """从网络相机获取一帧图像"""
        try:
            import mvsdk
            
            hCamera = camera_info['handle']
            pFrameBuffer = camera_info['buffer']
            
            # 获取图像数据
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 100)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
            
            # 转换为OpenCV格式
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            
            if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8:
                frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1))
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 3))

            # 翻转修正
            if self.network_flip_mode != 'none':
                if self.network_flip_mode == 'vertical':
                    frame = cv2.flip(frame, 0)
                elif self.network_flip_mode == 'horizontal':
                    frame = cv2.flip(frame, 1)
                elif self.network_flip_mode == 'both':  # 180度翻转
                    frame = cv2.flip(frame, -1)

            # 如果硬件白平衡不可用或关闭且需要软件补偿
            if (not camera_info.get('awb_on', False)) and self.software_wb_backup:
                frame = self.software_white_balance(frame)
            
            return frame
            
        except Exception as e:
            # 网络相机偶尔会超时，这是正常的
            return None
    
    def detect_pink_spray_bottle(self, frame):
        """检测粉色喷水壶；仅返回 left/right，center 返回 None 不触发切换"""
        if frame is None:
            return None, None
        
        # 使用YOLO进行检测
        results = self.model(frame, conf=self.detection_confidence, verbose=False)
        
        best_detection = None
        best_score = 0
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                
                # 检查是否为目标类别
                if class_name in self.target_classes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 检查是否为粉色
                    pink_score = self.calculate_pink_score(frame, int(x1), int(y1), int(x2), int(y2))
                    
                    # 综合评分 (检测置信度 + 粉色评分)
                    total_score = confidence * 0.7 + pink_score * 0.3
                    
                    if total_score > best_score and pink_score > 0.15:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        best_detection = {
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'center': (center_x, center_y),
                            'confidence': confidence,
                            'pink_score': pink_score,
                            'total_score': total_score,
                            'class': class_name
                        }
                        best_score = total_score
        
        if best_detection:
            # 判断位置
            frame_width = frame.shape[1]
            object_center_x = best_detection['center'][0]
            
            left_boundary = frame_width * self.position_threshold
            right_boundary = frame_width * (1 - self.position_threshold)
            
            if object_center_x < left_boundary:
                position = "left"
            elif object_center_x > right_boundary:
                position = "right"
            else:
                position = None  # 中间不触发切换
                
            return best_detection, position
        
        return None, None
    
    def calculate_pink_score(self, frame, x1, y1, x2, y2):
        """计算粉色评分"""
        try:
            # 提取目标区域
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return 0.0
            
            # 转换到HSV色彩空间
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 定义粉色的HSV范围（更宽松的范围）
            pink_ranges = [
                (np.array([140, 20, 40]), np.array([180, 255, 255])),  # 紫红色
                (np.array([0, 20, 40]), np.array([20, 255, 255])),     # 红粉色
                (np.array([160, 20, 50]), np.array([180, 255, 255])),  # 浅粉色
            ]
            
            total_pink_pixels = 0
            total_pixels = roi.shape[0] * roi.shape[1]
            
            for lower, upper in pink_ranges:
                mask = cv2.inRange(hsv, lower, upper)
                total_pink_pixels += cv2.countNonZero(mask)
            
            pink_ratio = min(total_pink_pixels / total_pixels, 1.0)
            return pink_ratio
            
        except Exception:
            return 0.0
    
    def should_switch_camera(self, position):
        """判断是否应该切换网络相机 (仅处理 left/right)"""
        current_time = time.time()
        if position not in ("left", "right"):
            return False
        
        # 检查冷却时间
        if current_time - self.last_switch_time < self.switch_cooldown:
            return False
        
        # 检查位置稳定性
        if position != self.last_position:
            self.last_detection_time = current_time
            self.last_position = position
            return False
        
        # 位置需要稳定一段时间才切换
        if current_time - self.last_detection_time < self.detection_stable_time:
            return False
        
        return True
    
    def switch_camera(self, target_camera, reason=""):
        """切换网络相机窗口"""
        with self.lock:
            if target_camera != self.current_display:
                self.current_display = target_camera
                self.last_switch_time = time.time()
                self.switch_count += 1
                print(f"📷 切换到: {target_camera} {reason}")

    # ---------------- 颜色与翻转辅助 -----------------
    def cycle_flip_mode(self):
        order = ['none', 'vertical', 'horizontal', 'both']
        idx = order.index(self.network_flip_mode) if self.network_flip_mode in order else 0
        self.network_flip_mode = order[(idx + 1) % len(order)]
        print(f"🔄 网络相机翻转模式 => {self.network_flip_mode}")

    def toggle_auto_wb(self):
        self.enable_auto_wb = not self.enable_auto_wb
        print(f"🎛️ 自动白平衡 => {'开启' if self.enable_auto_wb else '关闭'}")
        # 对所有网络相机应用
        try:
            import mvsdk
            for cam in self.network_cameras:
                if cam.get('wb_supported', False):
                    try:
                        mvsdk.CameraSetWbMode(cam['handle'], 1 if self.enable_auto_wb else 0)
                        cam['awb_on'] = self.enable_auto_wb
                        if self.enable_auto_wb:
                            mvsdk.CameraSetOnceWB(cam['handle'])
                    except Exception:
                        pass
        except Exception:
            pass

    def once_wb(self):
        try:
            import mvsdk
            for cam in self.network_cameras:
                if cam.get('wb_supported', False):
                    try:
                        mvsdk.CameraSetOnceWB(cam['handle'])
                        print(f"⚙️ 相机 {cam['id']} 进行一次白平衡")
                    except Exception:
                        pass
        except Exception:
            pass

    def software_white_balance(self, frame):
        # 简单灰世界白平衡 (防止黄色/绿色偏色)
        try:
            b, g, r = cv2.split(frame)
            mb, mg, mr = np.mean(b), np.mean(g), np.mean(r)
            if min(mb, mg, mr) < 1e-3:
                return frame
            k = (mb + mg + mr) / 3.0
            b = np.clip(b * (k / mb), 0, 255)
            g = np.clip(g * (k / mg), 0, 255)
            r = np.clip(r * (k / mr), 0, 255)
            return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])
        except Exception:
            return frame
    
    def detection_loop(self):
        """检测线程主循环"""
        print("🔍 检测线程启动")
        last_detection_result = None
        
        while self.running:
            if self.usb_camera is None:
                time.sleep(0.1)
                continue
                
            # 从USB摄像头读取帧
            try:
                ret, frame = self.usb_camera.read()
                if not ret:
                    print("⚠️ USB摄像头读取失败，尝试重新连接...")
                    if self.reconnect_usb_camera():
                        print("✓ USB摄像头重连成功")
                        continue
                    else:
                        print("❌ USB摄像头重连失败，等待重试...")
                        time.sleep(2.0)
                        continue
            except cv2.error as e:
                print(f"⚠️ USB摄像头读取异常: {e}")
                print("🔄 尝试重新连接USB摄像头...")
                if self.reconnect_usb_camera():
                    print("✓ USB摄像头重连成功")
                    continue
                else:
                    print("❌ USB摄像头重连失败，等待重试...")
                    time.sleep(2.0)
                    continue
            except Exception as e:
                print(f"⚠️ 摄像头异常: {e}")
                print("🔄 尝试重新连接USB摄像头...")
                if self.reconnect_usb_camera():
                    print("✓ USB摄像头重连成功")
                    continue
                else:
                    print("❌ USB摄像头重连失败，等待重试...")
                    time.sleep(2.0)
                    continue
            
            # 检测粉色喷水壶
            detection, position = self.detect_pink_spray_bottle(frame)
            
            # 检测状态稳定性处理
            current_has_detection = detection is not None and position in ("left", "right")
            
            if current_has_detection:
                self.current_stable_frames += 1
                self.current_no_detect_frames = 0
                
                # 连续检测到足够帧数，认为稳定检测到
                if self.current_stable_frames >= self.detection_stable_frames:
                    if not self.stable_detection_state:
                        self.stable_detection_state = True
                        print(f"✅ 稳定检测到目标 -> 位置: {position}")
                    
                    self.detection_count += 1
                    
                    # 判断是否需要切换相机
                    if self.should_switch_camera(position):
                        if position == "left" and len(self.network_cameras) > 0:
                            self.switch_camera("network_0", f"(检测到粉色{detection['class']}在左侧)")
                        elif position == "right" and len(self.network_cameras) > 1:
                            self.switch_camera("network_1", f"(检测到粉色{detection['class']}在右侧)")
                    
                    last_detection_result = (detection, position)
            else:
                self.current_no_detect_frames += 1
                self.current_stable_frames = 0
                
                # 连续检测不到足够帧数，认为稳定丢失
                if self.current_no_detect_frames >= self.no_detection_frames:
                    if self.stable_detection_state:
                        self.stable_detection_state = False
                        print("ℹ️ 稳定丢失目标")
            
            # 在USB摄像头画面上绘制检测结果（用于调试）
            if detection:
                self.draw_detection_info(frame, detection, position)
            
            # 缩放缓存供主窗口画中画
            try:
                self.last_usb_display = cv2.resize(frame, (320, 240))
            except Exception:
                pass
            
            # 更新共享检测状态（供主窗口显示）
            self.last_detect_found = self.stable_detection_state
            self.last_detect_position = position if self.stable_detection_state else None
            
            time.sleep(0.03)  # ~30fps
    
    def draw_detection_info(self, frame, detection, position):
        """在帧上绘制检测信息"""
        bbox = detection['bbox']
        
        # 根据位置选择颜色
        if position == "left":
            color = (255, 0, 0)  # 蓝色
        elif position == "right":
            color = (0, 255, 255)  # 黄色
        else:
            color = (0, 255, 0)  # 绿色
        
        # 绘制边界框
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # 绘制标签
        label = f"{detection['class']} ({position})"
        label += f" P:{detection['pink_score']:.2f}"
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 绘制区域分界线
        h, w = frame.shape[:2]
        left_line = int(w * self.position_threshold)
        right_line = int(w * (1 - self.position_threshold))
        
        cv2.line(frame, (left_line, 0), (left_line, h), (255, 255, 255), 2)
        cv2.line(frame, (right_line, 0), (right_line, h), (255, 255, 255), 2)
        
        cv2.putText(frame, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "RIGHT", (w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def display_loop(self):
        """显示线程主循环 (单窗口 + USB画中画)"""
        print("📺 显示线程启动 (单窗口 + USB画中画)")
        MAIN_WINDOW = "智能摄像头系统"
        cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
        
        # 灰窗首帧黑底
        black_frame = np.zeros((830, 1000, 3), dtype=np.uint8)
        self.draw_text_cn(black_frame, "等待网络相机画面...", (400, 400), size=32, color=(200,200,200))
        cv2.imshow(MAIN_WINDOW, black_frame)
        cv2.waitKey(1)
        
        # 移除轨迹条，改用键盘控制
        last_mode = None
        
        while self.running:
            net_frame = None
            usb_frame = None
            with self.lock:
                current_mode = self.current_display
            
            # USB 永远读取
            if self.usb_camera:
                ret_u, usb_frame = self.usb_camera.read()
                if not ret_u:
                    usb_frame = None
                    
            # 网络相机按当前模式读取
            if current_mode == "network_0" and len(self.network_cameras) > 0:
                net_frame = self.get_network_camera_frame(self.network_cameras[0])
            elif current_mode == "network_1" and len(self.network_cameras) > 1:
                net_frame = self.get_network_camera_frame(self.network_cameras[1])
            
            # 显示主画面
            if net_frame is not None:
                display_frame = cv2.resize(net_frame, (1000, 750))
                params = self.camera_params.get(current_mode, {'zoom':0,'sharpen':0})
                
                # 数字变焦
                if params['zoom'] > 0:
                    zp = min(max(params['zoom'], 0), 60)
                    h, w = display_frame.shape[:2]
                    crop_ratio = 1 - zp / 100.0
                    cw = max(int(w * crop_ratio), 50)
                    ch = max(int(h * crop_ratio), 50)
                    x1 = (w - cw) // 2
                    y1 = (h - ch) // 2
                    crop = display_frame[y1:y1+ch, x1:x1+cw]
                    display_frame = cv2.resize(crop, (w, h))
                
                # 锐化
                if params['sharpen'] > 0:
                    level = min(params['sharpen'], 5)
                    alpha = 1.0 + 0.4 * level
                    blurred = cv2.GaussianBlur(display_frame, (0, 0), sigmaX=1.2 + 0.3 * level)
                    display_frame = cv2.addWeighted(display_frame, alpha, blurred, -(alpha - 1), 0)
                
                # 信息条
                info_bg = np.zeros((80, display_frame.shape[1], 3), dtype=np.uint8)
                info_bg[:] = (0, 0, 0)
                status_text = f"网络显示:{current_mode} 检测:{self.detection_count} 切换:{self.switch_count}"
                help_text = "手动: 1=网络0 2=网络1 | U=USB画中画开关 | +/-=缩放 [/]=锐化 | Q=退出"
                extra = f"翻转:{self.network_flip_mode} 自动WB:{'开' if self.enable_auto_wb else '关'} F翻转 W自动 O一次 B软WB Z={params['zoom']}% S锐化:{params['sharpen']}"
                self.draw_text_cn(info_bg, status_text, (10, 8), size=24, color=(0, 255, 255))
                self.draw_text_cn(info_bg, help_text, (10, 35), size=20, color=(255, 255, 255))
                self.draw_text_cn(info_bg, extra, (10, 60), size=18, color=(200, 200, 200))
                
                detect_line = f"目标: {'有' if self.last_detect_found else '无'}"
                if self.last_detect_found and self.last_detect_position:
                    detect_line += f"({self.last_detect_position})"
                self.draw_text_cn(info_bg, detect_line, (600, 8), size=20, color=(0, 255, 0) if self.last_detect_found else (0, 0, 255))
                
                # 画中画
                if self.show_usb_pip and self.last_usb_display is not None:
                    pip = self.last_usb_display.copy()
                    ph, pw = pip.shape[:2]
                    cv2.rectangle(pip, (0,0),(pw-1,ph-1),(255,255,255),2)
                    dh, dw = display_frame.shape[:2]
                    x1 = dw - pw - 10
                    y1 = dh - ph - 10
                    roi = display_frame[y1:y1+ph, x1:x1+pw]
                    if roi.shape[:2] == pip.shape[:2]:
                        display_frame[y1:y1+ph, x1:x1+pw] = pip
                
                final_frame = np.vstack([info_bg, display_frame])
                cv2.imshow(MAIN_WINDOW, final_frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("🛑 收到退出信号")
                self.running = False
                break
            elif key == ord('1') and len(self.network_cameras) > 0:
                self.switch_camera("network_0", "(手动切换)")
            elif key == ord('2') and len(self.network_cameras) > 1:
                self.switch_camera("network_1", "(手动切换)")
            elif key in (ord('f'), ord('F')):
                self.cycle_flip_mode()
                self.save_config()
            elif key in (ord('w'), ord('W')):
                self.toggle_auto_wb()
                self.save_config()
            elif key in (ord('o'), ord('O')):
                self.once_wb()
            elif key in (ord('b'), ord('B')):
                self.software_wb_backup = not self.software_wb_backup
                print(f"🧪 软件白平衡备用 => {'启用' if self.software_wb_backup else '停用'}")
            elif key in (ord('u'), ord('U')):
                self.show_usb_pip = not self.show_usb_pip
                print(f"🪟 USB画中画 => {'显示' if self.show_usb_pip else '隐藏'}")
                self.save_config()
            elif key in (ord('s'), ord('S')):
                self.save_config()
                print("💾 已手动保存配置")
            elif key == ord('+') or key == ord('='):  # 增加缩放
                with self.lock:
                    mode = self.current_display
                    current_zoom = self.camera_params[mode]['zoom']
                    self.camera_params[mode]['zoom'] = min(current_zoom + 5, 60)
                    print(f"🔍 {mode} 缩放: {self.camera_params[mode]['zoom']}%")
                self.save_config()
            elif key == ord('-') or key == ord('_'):  # 减少缩放
                with self.lock:
                    mode = self.current_display
                    current_zoom = self.camera_params[mode]['zoom']
                    self.camera_params[mode]['zoom'] = max(current_zoom - 5, 0)
                    print(f"🔍 {mode} 缩放: {self.camera_params[mode]['zoom']}%")
                self.save_config()
            elif key == ord('['):  # 增加锐化
                with self.lock:
                    mode = self.current_display
                    current_sharpen = self.camera_params[mode]['sharpen']
                    self.camera_params[mode]['sharpen'] = min(current_sharpen + 1, 5)
                    print(f"✨ {mode} 锐化: {self.camera_params[mode]['sharpen']}")
                self.save_config()
            elif key == ord(']'):  # 减少锐化
                with self.lock:
                    mode = self.current_display
                    current_sharpen = self.camera_params[mode]['sharpen']
                    self.camera_params[mode]['sharpen'] = max(current_sharpen - 1, 0)
                    print(f"✨ {mode} 锐化: {self.camera_params[mode]['sharpen']}")
                self.save_config()
            
            time.sleep(0.03)
            
            time.sleep(0.03)
        
        cv2.destroyAllWindows()
    
    def cleanup(self):
        """清理资源"""
        print("🧹 正在清理资源...")
        
        # 关闭USB摄像头
        if self.usb_camera:
            self.usb_camera.release()
            print("✓ USB摄像头已关闭")
        
        # 关闭网络相机
        if self.network_cameras:
            try:
                import mvsdk
                for camera_info in self.network_cameras:
                    try:
                        mvsdk.CameraUnInit(camera_info['handle'])
                        mvsdk.CameraAlignFree(camera_info['buffer'])
                    except Exception as e:
                        print(f"⚠️  关闭网络相机失败: {e}")
                print(f"✓ {len(self.network_cameras)}个网络相机已关闭")
            except ImportError:
                pass
        
        cv2.destroyAllWindows()
        print("✅ 资源清理完成")
        print(f"📊 运行统计: 检测{self.detection_count}次, 切换{self.switch_count}次")

def main():
    """主函数"""
    try:
        system = SmartCameraSystem()
        system.run()
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()