"""
process_file接口：

立即返回task_id，表示任务创建成功。

将实际处理放入后台任务，使用BackgroundTasks。

新增get_task_status接口：

根据task_id查询任务状态。

返回任务状态（pending、running、completed、failed）和结果（如果有）。
"""
from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import requests
from pathlib import Path
import subprocess
import json
import uuid

app = FastAPI()

# 根目录
BASE_SAVE_DIR = Path("./saved_files")
BASE_RESULTS_DIR = Path("./results")

# 模型路径
MODEL_PATH = Path("./hf_model")

# 确保根目录存在
BASE_SAVE_DIR.mkdir(exist_ok=True)
BASE_RESULTS_DIR.mkdir(exist_ok=True)

# 任务状态存储
task_status = {}


def download_file(url: str, save_dir: Path) -> Path:
    local_filename = save_dir / url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def run_recognition(input_path: Path, save_dir: Path) -> None:
    cmd = [
        "python", "demo_page_hf.py",
        "--model_path", str(MODEL_PATH),
        "--input_path", str(input_path),
        "--save_dir", str(save_dir)
    ]
    subprocess.run(cmd, check=True)


def process_file_task(
        task_id: str,
        file_url: str,
        output_format: str
):
    task_status[task_id] = {"status": "running", "result": None}

    # 创建任务对应的保存目录
    task_save_dir = BASE_SAVE_DIR / task_id
    task_results_dir = BASE_RESULTS_DIR / task_id
    task_save_dir.mkdir(parents=True, exist_ok=True)
    task_results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 判断file_url是在线文件还是本地路径
        if file_url.lower().startswith("http://") or file_url.lower().startswith("https://"):
            # 在线文件，下载保存
            try:
                saved_path = download_file(file_url, task_save_dir)
            except Exception as e:
                task_status[task_id] = {"status": "failed", "error": f"下载文件失败: {str(e)}"}
                return
        else:
            # 本地路径，直接使用
            saved_path = Path(file_url)
            if not saved_path.exists():
                task_status[task_id] = {"status": "failed", "error": "本地文件不存在"}
                return
            if not saved_path.is_absolute():
                saved_path = saved_path.resolve()

        # 调用识别脚本，指定结果保存目录为task_results_dir
        try:
            run_recognition(saved_path, task_results_dir)
        except subprocess.CalledProcessError as e:
            task_status[task_id] = {"status": "failed", "error": f"识别过程失败: {str(e)}"}
            return

        if output_format == "markdown":
            md_dir = task_results_dir / "markdown"
            if not md_dir.exists() or not md_dir.is_dir():
                task_status[task_id] = {"status": "failed", "error": "未找到Markdown结果文件夹"}
                return

            result_list = []
            for file_path in md_dir.glob("*.md"):
                content = file_path.read_text(encoding="utf-8")
                result_list.append({file_path.name: content})

            task_status[task_id] = {"status": "completed", "result": result_list}

        elif output_format == "json":
            json_dir = task_results_dir / "recognition_json"
            if not json_dir.exists() or not json_dir.is_dir():
                task_status[task_id] = {"status": "failed", "error": "未找到JSON结果文件夹"}
                return

            result_list = []
            for file_path in json_dir.glob("*.json"):
                try:
                    content = json.loads(file_path.read_text(encoding="utf-8"))
                    result_list.append({file_path.name: content})
                except json.JSONDecodeError:
                    # 文件不是合法json时跳过或返回错误，这里选择跳过
                    continue

            task_status[task_id] = {"status": "completed", "result": result_list}

        else:
            task_status[task_id] = {"status": "failed", "error": "无效的输出格式参数"}

    except Exception as e:
        task_status[task_id] = {"status": "failed", "error": str(e)}


@app.post("/process_file")
async def process_file(
        file_url: str = Form(...),
        output_format: str = Form("markdown"),  # 支持"markdown"或"json"
        background_tasks: BackgroundTasks = None
):
    # 生成唯一任务ID
    task_id = str(uuid.uuid4())
    task_status[task_id] = {"status": "pending"}

    # 添加后台任务
    background_tasks.add_task(process_file_task, task_id, file_url, output_format)

    # 立即返回task_id
    return JSONResponse(content={"task_id": task_id})


@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in task_status:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    return JSONResponse(content=task_status[task_id])
