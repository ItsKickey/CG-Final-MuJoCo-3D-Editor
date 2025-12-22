import glfw
from OpenGL.GL import glGetString, GL_VERSION, GL_RENDERER, GL_VENDOR

def check_opengl_version():
    # 初始化 GLFW
    if not glfw.init():
        print("Failed to initialize GLFW")
        return

    # 建立一個隱藏的視窗來獲取 Context
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(640, 480, "OpenGL Check", None, None)
    
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        return

    # 設定 Context
    glfw.make_context_current(window)

    # 獲取資訊
    version = glGetString(GL_VERSION)
    renderer = glGetString(GL_RENDERER)
    vendor = glGetString(GL_VENDOR)

    print("-" * 30)
    print(f"OpenGL Version:  {version.decode('utf-8')}")
    print(f"Renderer:        {renderer.decode('utf-8')}")
    print(f"Vendor:          {vendor.decode('utf-8')}")
    print("-" * 30)

    # 判斷是否為常見的微軟通用驅動 (導致錯誤的主因)
    if b"GDI Generic" in renderer or version.startswith(b"1.1"):
        print("⚠️ 警告: 偵測到 Windows 預設通用驅動 (GDI Generic)。")
        print("   這通常發生在遠端桌面 (RDP) 或未安裝顯卡驅動的環境。")
        print("   此版本不支援 Framebuffer Object (FBO)，會導致 MuJoCo 崩潰。")

    glfw.terminate()

if __name__ == "__main__":
    check_opengl_version()