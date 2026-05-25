
import random
import shutil
from pathlib import Path
log = print
SOURCE_DIR = "data/resampled"           # откуда брать файлы
DEST_DIR = "data/resampled_split"     # куда сохранять
TRAIN_RATIO = 0.8             # 80%
VAL_RATIO = 0.1                # 10%
TEST_RATIO = 0.1               # 10%
RANDOM_SEED = 42               # для воспроизводимости
MOVE_FILES = True             # True = перемещать, False = копировать
# ===============================

random.seed(RANDOM_SEED)

source_path = Path(SOURCE_DIR)
dest_path = Path(DEST_DIR)

if not source_path.exists():
    log(f"Error: folder {SOURCE_DIR} wasn't found")
    exit(1)

files = [f for f in source_path.iterdir() if f.is_file()]
if not files:
    log(f"Error: {SOURCE_DIR} has no files")
    exit(1)

random.shuffle(files)

total = len(files)
train_size = int(total * TRAIN_RATIO)
val_size = int(total * VAL_RATIO)

train_files = files[:train_size]
val_files = files[train_size:train_size + val_size]
test_files = files[train_size + val_size:]

for folder in ['training', 'validation', 'test']:
    (dest_path / folder).mkdir(parents=True, exist_ok=True)

def process_files(file_list, target_folder): # type: ignore
    target = dest_path / target_folder
    for f in file_list:
        if MOVE_FILES:
            shutil.move(str(f), str(target / f.name))
        else:
            shutil.copy2(f, target / f.name)

process_files(train_files, 'training')
process_files(val_files, 'validation')
process_files(test_files, 'test')

log(f"Total: {total} files")
log(f"Training:   {len(train_files)} ({len(train_files)/total*100:.1f}%)")
log(f"Validation: {len(val_files)} ({len(val_files)/total*100:.1f}%)")
log(f"Test:       {len(test_files)} ({len(test_files)/total*100:.1f}%)")
log(f"Files {'moved' if MOVE_FILES else 'copied'} into {DEST_DIR}")