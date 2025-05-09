import os
from django.conf import settings

def clear_media_folder():
    media_dir = settings.MEDIA_ROOT
    for filename in os.listdir(media_dir):
        file_path = os.path.join(media_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")