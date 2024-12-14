#### 인공지능 모델

#### env
- base python : 3.9

#### test case
- 가상환경에서 다음 명령어 실행
```
uvicorn _serve_app:app --host 0.0.0.0 --port 8000
```
- 병렬처리 성능 테스트(Apache Bench)
```
ab -n 1000 -c 10 -p request.json -T application/json http://localhost:8000/execute
```
- _serve_app_ray.py 호출 명령어
```
uvicorn _serve_app_ray:app --host 0.0.0.0 --port 8000
curl -X POST "http://localhost:8000/execute" \
     -H "Content-Type: application/json" \
     -d '{"data": [1, 2, 3, 4, 5]}'

```

#### command
- preprocess
```
python main.py --data_dir ./_source/data --train_data train.csv --preprocess
```
- scale
```
python main.py --data_dir ./_source/data --train_data train.csv --scale
```
- train
```
python main.py --data_dir ./_source/data --train_data train.csv --train --epochs 100
```
- test
```
python main.py --data_dir ./_source/data --test_data test.csv --test
```
- predict
```
python main.py --data_dir ./_source/data --predict_data predict.csv --predict
```
- optimize
```
python main.py --data_dir ./_source/data --train_data train.csv --test_data test.csv --optimize
```