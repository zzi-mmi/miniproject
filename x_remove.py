# 필요한 라이브러리 및 모듈 import
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from typing import List
import shutil
import subprocess
from insightface.app import FaceAnalysis

# FastAPI 애플리케이션 객체 생성
app = FastAPI()

# 얼굴 분석 객체 초기화
face_analysis = FaceAnalysis()
face_analysis.prepare(ctx_id=-1, det_size=(640, 640))

# 이미지에서 얼굴 특징 벡터를 추출하는 함수 정의
def get_face_embedding(img):
    faces = face_analysis.get(img)
    if faces:
        return faces[0].normed_embedding
    else:
        return None

# 학습된 이미지 폴더에서 평균 얼굴 특징 벡터를 계산하는 함수 정의
def get_average_face_embedding(train_folder_path):
    embeddings = []
    for filename in os.listdir(train_folder_path):
        img_path = os.path.join(train_folder_path, filename)
        img = cv2.imread(img_path)
        embedding = get_face_embedding(img)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            print(f"이미지 {img_path}의 얼굴 임베딩 추출에 실패했습니다.")
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        # 얼굴 특징 벡터가 없을 경우 기본값으로 512차원의 영벡터를 반환합니다.
        return np.zeros(512)

# 분류된 이미지를 이동시키는 함수 정의
def move_image_to_folder(file_data, folder_path, file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    dest_path = os.path.join(folder_path, file_name)
    with open(dest_path, "wb") as f:
        f.write(file_data)
    print(f"이미지 {file_name}를 {folder_path}로 이동했습니다.")

# 분류된 이미지를 삭제하는 함수 정의
def delete_classified_images(folder_path):
    if os.path.exists(folder_path):
        # 폴더 내의 모든 파일을 삭제하고 폴더 자체를 삭제합니다.
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
        shutil.rmtree(folder_path)
        print(f"폴더 {folder_path}와 해당 내용물이 삭제되었습니다.")
        return True
    else:
        print(f"폴더 {folder_path}가 존재하지 않습니다.")
        return False

# 이미지를 업로드하고 얼굴을 분석하는 엔드포인트 정의
@app.post("/uploadimages/")
async def create_upload_images(files: List[UploadFile] = File(...)):
    train_folder_path = "/test1"  # 학습된 이미지가 있는 폴더의 경로
    target_embedding = get_average_face_embedding(train_folder_path)
    
    # 모든 업로드된 파일을 처리합니다.
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 얼굴 특징 벡터 추출
        embedding = get_face_embedding(img)
        
        if embedding is not None:
            # 유사성 계산
            similarity = np.dot(embedding, target_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(target_embedding))
            # 분류 기준에 따라 분류 및 이동을 수행하는 코드 작성
            similarity_threshold = 0.6
            if similarity > similarity_threshold:
                # 분류된 이미지를 이동합니다.
                move_image_to_folder(contents, "/test/test1", file.filename)  # 파일 데이터와 파일 이름을 이동시킬 폴더 경로 지정
            else:
                # 분류되지 않은 이미지를 이동합니다.
                move_image_to_folder(contents, "/test/test2", file.filename)  # 파일 데이터와 파일 이름을 이동시킬 폴더 경로 지정
        else:
            print(f"이미지 {file.filename}의 얼굴 임베딩 추출에 실패했습니다.")

    # 업로드 결과를 HTML 형식으로 반환합니다.
    return HTMLResponse(content="""
    <html>
    <head>
        <title>X-Remove</title>
    </head>
    <body align="center">
        <h1>X 분류 완료</h1>
        <p>X를 성공적으로 분류 및 이동했습니다.</p>
        <form action="/deleteclassifiedimages/" method="post">
            <button type="submit">분류된 X 바로 삭제</button>
        </form>
        <form action="/openclassifiedfolder/" method="post">
            <button type="submit">분류된 X 파일 확인</button>
        </form>
    </body>
    </html>
    """, status_code=200)

# 분류된 이미지를 삭제하는 엔드포인트 정의
@app.post("/deleteclassifiedimages/")
async def delete_classified_images_endpoint():
    if delete_classified_images("/test/test1"):
        print("분류된 X가 삭제되었습니다.")
    else:
        print("분류된 X가 삭제되지 않았습니다.")

    # 삭제 결과를 HTML 형식으로 반환합니다.
    confirmation_message = "분류된 X가 삭제되었습니다."
    back_button = """
    <form action="/" method="get">
        <button type="submit">처음으로</button>
    </form>
    """
    return HTMLResponse(content=f"<h1>{confirmation_message}</h1>{back_button}", status_code=200)

# 분류된 이미지 폴더를 열어주는 엔드포인트 정의
@app.post("/openclassifiedfolder/")
async def open_classified_folder_endpoint():
    classified_folder_path = "/test/test1"  # 분류된 이미지가 저장된 폴더의 경로
    subprocess.Popen(f'explorer "{os.path.realpath(classified_folder_path)}"')  # 폴더 열기
    print(f"분류된 이미지가 저장된 폴더인 {classified_folder_path}를 열었습니다.")

    # 폴더를 열었다는 확인 메시지와 삭제 버튼을 HTML 형식으로 반환합니다.
    confirmation_message = "분류된 이미지가 저장된 폴더가 열렸습니다."
    delete_button = """
    <form action="/deleteclassifiedimages/" method="post">
        <button type="submit">삭제하기</button>
    </form>
    """
    return HTMLResponse(content=f"<h1>{confirmation_message}</h1>{delete_button}", status_code=200)

# HTML 폼을 렌더링하는 엔드포인트 정의
@app.get("/", response_class=HTMLResponse)
async def read_form():
    return """
    <html>
    <head>
        <title>X-Remove</title>
    </head>
    <body align="center">
        <h1>X-Remove</h1>
        <h1>당신의 X를 지워드립니다</h1>
        <form action="/uploadimages/" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <button type="submit">업로드</button>
        </form>
    </body>
    </html>
    """
