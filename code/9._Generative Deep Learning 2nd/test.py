import tensorflow as tf

# GPU 디바이스 목록 가져오기
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    print("사용 가능한 GPU 디바이스:")
    for gpu in gpus:
        print("디바이스 이름:", gpu.name)
        print("메모리 제한:", gpu.memory_limit)
else:
    print("사용 가능한 GPU 디바이스가 없습니다.")
    