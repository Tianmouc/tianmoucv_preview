#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import socket
import time
from datetime import datetime

# ---------- 参照你代码里的 UDP 控制协议 ----------
class UdpCameraController:
    def __init__(self, ip: str, port: int, log_fn):
        self.ip = ip
        self.port = port
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.log = log_fn

    def update_addr(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.addr = (ip, port)
        self.log(f"[SYS] 目标地址更新为 {self.addr}")

    def send(self, msg: str):
        data = msg.encode("utf-8")
        try:
            self.sock.sendto(data, self.addr)
            self.log(f"[UDP] -> {self.addr}: {msg}")
        except Exception as e:
            self.log(f"[ERR] 发送失败: {e}")

    # 与你现有的控制指令保持一致
    def set_aop_exposure_gain(self, exp: int, gain: int):
        self.send(f"set_aop_exposure:{exp},{gain}")

    def set_cop_exposure_gain(self, exp: int, gain: int):
        self.send(f"set_cop_exposure:{exp},{gain}")

    def set_save_addr(self, path: str):
        self.send("set_addr:" + path)

    def start_camera(self):
        self.send("start_camera")

    def start_record(self):
        self.send("start_record")

    def stop_record(self):
        self.send("stop_record")

    def stop_camera(self):
        self.send("stop_camera")


# ---------- Tk GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tianmouc 相机 UDP 控制台 (简易版)")
        self.geometry("780x560")
        self.minsize(740, 520)

        # 主题
        try:
            self.call("tk", "scaling", 1.25)
            style = ttk.Style()
            style.theme_use("clam")
        except Exception:
            pass

        # 状态
        self.ip_var = tk.StringVar(value="10.42.0.1")
        self.port_var = tk.StringVar(value="8889")

        self.aop_exp_var = tk.StringVar(value="1240")
        self.aop_gain_var = tk.StringVar(value="1")
        self.cop_exp_var = tk.StringVar(value="14520")
        self.cop_gain_var = tk.StringVar(value="1")

        self.save_path_var = tk.StringVar(value="/home/nvidia/UAV1016/1")

        # 日志
        self.log_text = None

        # 控制器
        self.ctrl = UdpCameraController(self.ip_var.get(), int(self.port_var.get()), self._log)

        # UI
        self._build_ui()

        # 绑定回车快捷操作
        self.bind("<Return>", lambda e: None)  # 防止多控件抢占回车
        self.bind("<Control-Return>", lambda e: self.on_start_record())

    # ---------- UI 构建 ----------
    def _build_ui(self):
        root = ttk.Frame(self, padding=(10, 10, 10, 10))
        root.pack(fill=tk.BOTH, expand=True)

        # 连接配置
        conn = ttk.LabelFrame(root, text="连接")
        conn.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(conn, text="IP:").grid(row=0, column=0, padx=(10, 4), pady=8, sticky="w")
        ip_entry = ttk.Entry(conn, textvariable=self.ip_var, width=18)
        ip_entry.grid(row=0, column=1, padx=(0, 10), pady=8)

        ttk.Label(conn, text="端口:").grid(row=0, column=2, padx=(10, 4), pady=8, sticky="w")
        port_entry = ttk.Entry(conn, textvariable=self.port_var, width=8)
        port_entry.grid(row=0, column=3, padx=(0, 10), pady=8)

        ttk.Button(conn, text="应用地址", command=self.on_apply_addr).grid(row=0, column=4, padx=(10, 10), pady=8)

        # 参数区
        param = ttk.LabelFrame(root, text="参数")
        param.pack(fill=tk.X, pady=(0, 8))

        # AOP
        ttk.Label(param, text="AOP 曝光").grid(row=0, column=0, padx=10, pady=(10, 4), sticky="w")
        ttk.Entry(param, textvariable=self.aop_exp_var, width=10).grid(row=0, column=1, padx=(0, 10), pady=(10, 4))
        ttk.Label(param, text="AOP 增益").grid(row=0, column=2, padx=10, pady=(10, 4), sticky="w")
        ttk.Entry(param, textvariable=self.aop_gain_var, width=10).grid(row=0, column=3, padx=(0, 10), pady=(10, 4))
        ttk.Button(param, text="发送 AOP 参数", command=self.on_send_aop).grid(row=0, column=4, padx=10, pady=(10, 4))

        # COP
        ttk.Label(param, text="COP 曝光").grid(row=1, column=0, padx=10, pady=4, sticky="w")
        ttk.Entry(param, textvariable=self.cop_exp_var, width=10).grid(row=1, column=1, padx=(0, 10), pady=4)
        ttk.Label(param, text="COP 增益").grid(row=1, column=2, padx=10, pady=4, sticky="w")
        ttk.Entry(param, textvariable=self.cop_gain_var, width=10).grid(row=1, column=3, padx=(0, 10), pady=4)
        ttk.Button(param, text="发送 COP 参数", command=self.on_send_cop).grid(row=1, column=4, padx=10, pady=4)

        # Save 路径
        ttk.Label(param, text="保存目录").grid(row=2, column=0, padx=10, pady=(4, 10), sticky="w")
        ttk.Entry(param, textvariable=self.save_path_var, width=48).grid(row=2, column=1, columnspan=3, padx=(0, 10), pady=(4, 10), sticky="we")
        ttk.Button(param, text="选择...", command=self.on_browse_dir).grid(row=2, column=4, padx=10, pady=(4, 10))
        ttk.Button(param, text="发送保存目录", command=self.on_send_save_path).grid(row=2, column=5, padx=(0, 10), pady=(4, 10))

        # 控制区
        ctrl = ttk.LabelFrame(root, text="控制")
        ctrl.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(ctrl, text="启动相机", command=self.on_start_camera).grid(row=0, column=0, padx=10, pady=10, sticky="we")
        ttk.Button(ctrl, text="开始录制 (Ctrl+Enter)", command=self.on_start_record).grid(row=0, column=1, padx=10, pady=10, sticky="we")
        ttk.Button(ctrl, text="停止录制", command=self.on_stop_record).grid(row=0, column=2, padx=10, pady=10, sticky="we")
        ttk.Button(ctrl, text="停止相机", command=self.on_stop_camera).grid(row=0, column=3, padx=10, pady=10, sticky="we")

        for i in range(4):
            ctrl.grid_columnconfigure(i, weight=1)

        # 日志
        logf = ttk.LabelFrame(root, text="日志")
        logf.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(logf, height=14, wrap="word", state="disabled")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status = ttk.Label(root, textvariable=self.status_var, anchor="w")
        status.pack(fill=tk.X, pady=(4, 0))

    # ---------- 工具 ----------
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, line)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _get_ip_port(self):
        ip = self.ip_var.get().strip()
        port_s = self.port_var.get().strip()
        try:
            port = int(port_s)
            if not (0 < port < 65536):
                raise ValueError
        except Exception:
            raise ValueError("端口必须是 1~65535 的整数")
        # 粗略校验 IP
        try:
            socket.inet_aton(ip)
        except OSError:
            raise ValueError("IP 格式不正确")
        return ip, port

    def _get_int(self, var: tk.StringVar, name: str, minv=None, maxv=None):
        s = var.get().strip()
        try:
            v = int(s)
        except Exception:
            raise ValueError(f"{name} 必须是整数")
        if minv is not None and v < minv:
            raise ValueError(f"{name} 不能小于 {minv}")
        if maxv is not None and v > maxv:
            raise ValueError(f"{name} 不能大于 {maxv}")
        return v

    # ---------- 事件 ----------
    def on_apply_addr(self):
        try:
            ip, port = self._get_ip_port()
            self.ctrl.update_addr(ip, port)
            self.status_var.set(f"连接目标：{ip}:{port}")
        except Exception as e:
            messagebox.showerror("地址错误", str(e))

    def on_send_aop(self):
        try:
            exp = self._get_int(self.aop_exp_var, "AOP 曝光", 1)
            gain = self._get_int(self.aop_gain_var, "AOP 增益", 0)
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return
        self.ctrl.set_aop_exposure_gain(exp, gain)

    def on_send_cop(self):
        try:
            exp = self._get_int(self.cop_exp_var, "COP 曝光", 1)
            gain = self._get_int(self.cop_gain_var, "COP 增益", 0)
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return
        self.ctrl.set_cop_exposure_gain(exp, gain)

    def on_send_save_path(self):
        path = self.save_path_var.get().strip()
        if not path:
            messagebox.showerror("路径错误", "保存目录不能为空")
            return
        self.ctrl.set_save_addr(path)

    def on_browse_dir(self):
        d = filedialog.askdirectory(title="选择保存目录")
        if d:
            self.save_path_var.set(d)

    def on_start_camera(self):
        self.ctrl.start_camera()
        self.status_var.set("相机已启动（指令已发送）")

    def on_start_record(self):
        self.ctrl.start_record()
        self.status_var.set("开始录制（指令已发送）")

    def on_stop_record(self):
        self.ctrl.stop_record()
        self.status_var.set("停止录制（指令已发送）")

    def on_stop_camera(self):
        self.ctrl.stop_camera()
        self.status_var.set("相机已停止（指令已发送）")


if __name__ == "__main__":
    app = App()
    app.mainloop()
