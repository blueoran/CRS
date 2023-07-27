from toolbox import find_free_port, ArgsGeneralWrapper, chat_with_awesome
import gradio as gr
from theme import adjust_theme, advanced_css
import logging
import os
os.environ['no_proxy'] = '*'  # 避免代理网络产生意外污染

WEB_PORT = -1
AUTHENTICATION = None
CHATBOT_HEIGHT = 1145
CONCURRENT_COUNT = 10

PORT = find_free_port() if WEB_PORT <= 0 else WEB_PORT

title_html = f"<h1 align=\"center\">CRS 1.0.0</h1>"


# 做一些外观色彩上的调整
set_theme = adjust_theme()


def gr_L1(): return gr.Row().style()
def gr_L2(scale): return gr.Column(scale=scale)


cancel_handles = []
with gr.Blocks(title="CRS", theme=set_theme, analytics_enabled=False, css=advanced_css) as demo:
    gr.HTML(title_html)
    with gr_L1():
        with gr_L2(scale=2):
            chatbot = gr.Chatbot()
            chatbot.style(height=CHATBOT_HEIGHT)
            history = gr.State([])
        with gr_L2(scale=1):
            with gr.Accordion("Let's have a movie chat!", open=True) as area_input_primary:
                with gr.Row():
                    txt = gr.Textbox(show_label=False, placeholder="Input sentence here.").style(
                        container=False)
                with gr.Row():
                    submitBtn = gr.Button("提交", variant="primary")
                with gr.Row():
                    resetBtn = gr.Button("重置", variant="secondary")
                    resetBtn.style(size="sm")
                    stopBtn = gr.Button("停止", variant="secondary")
                    stopBtn.style(size="sm")
                with gr.Row():
                    status = gr.Markdown(
                        f"Tip: 按Enter提交, 按Shift+Enter换行。当前模型: GPT-3.5-turbo \n")

    # 整理反复出现的控件句柄组合
    input_combo = [txt, chatbot, history]
    output_combo = [chatbot, history, status]
    predict_args = dict(fn=ArgsGeneralWrapper(chat_with_awesome),
                        inputs=input_combo, outputs=output_combo)

    # 提交按钮、重置按钮
    cancel_handles.append(txt.submit(**predict_args))
    cancel_handles.append(submitBtn.click(**predict_args))
    resetBtn.click(lambda: ([], [], "已重置"), None, [chatbot, history, status])
    stopBtn.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)


def auto_opentab_delay():
    import threading
    import webbrowser
    import time
    print(f"如果浏览器没有自动打开，请复制并转到以下URL：")
    print(f"\t（亮色主题）: http://localhost:{PORT}")
    print(f"\t（暗色主题）: http://localhost:{PORT}/?__dark-theme=true")

    time.sleep(2)
    webbrowser.open_new_tab(f"http://localhost:{PORT}/?__dark-theme=true")


auto_opentab_delay()
demo.queue(concurrency_count=CONCURRENT_COUNT).launch(
    server_name="0.0.0.0", server_port=PORT, auth=AUTHENTICATION,share=True)
