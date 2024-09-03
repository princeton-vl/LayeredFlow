import requests
import argparse
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm
import os

website = 'https://layeredflow.cs.princeton.edu'

session = requests.Session()    
response = session.get(f'{website}/request_submit/')
csrf_token = session.cookies.get('csrftoken')
headers = {
    'X-CSRFToken': csrf_token,
    'Referer': f'{website}/request_submit/'
}

def request_verification(args, url=f"{website}/request_submit/"):
    data = {
        'email': args.email,
        'benchmark_name': args.benchmark,
        'method_name': args.method_name,
    }

    response = session.post(url, data=data, headers=headers)

    if response.status_code == 200:
        upload_id = response.json()['upload_id']
        print(f"Verification code sent to {args.email}.")
        print(f"Your submission id is {upload_id}.")
        return upload_id
    else:
        print("Failed to request verification:", response.text)
        exit(1)

def verify_code(upload_id, code, url=f"{website}/verify"):
    data = {
        'code': code,
    }
    response = session.post(f"{url}/{upload_id}/", data=data, headers=headers)
    if response.status_code == 200:
        return True
    else:
        print(response)
        exit(1)

def upload_file(upload_id, file_path, url=f"{website}/upload"):
    # Get files path
    file_parts = [os.path.join(file_path, f) for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    file_parts.sort()

    # Total size of all parts
    total_size = sum(os.path.getsize(part) for part in file_parts)
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Uploading")

    previous_uploaded_bytes = 0
    def progress_callback(monitor):
        progress_bar.update(monitor.bytes_read + previous_uploaded_bytes - progress_bar.n)

    # Upload each part
    for part in file_parts:
        with open(part, 'rb') as f:
            encoder = MultipartEncoder(
                fields={'file': (os.path.basename(part), f, 'application/gzip')}
            )
            monitor = MultipartEncoderMonitor(encoder, progress_callback)
            new_headers = headers.copy()
            new_headers['Content-Type'] = monitor.content_type
            response = session.post(f"{url}/{upload_id}/", data=monitor, headers=new_headers)

            # Close the part file after upload
            previous_uploaded_bytes += os.path.getsize(part)
            if response.status_code != 200:
                print("Failed to upload file:", response.text)
                progress_bar.close()
                exit(1)

    progress_bar.close()

    # Call finish_upload to combine all parts
    response = session.post(f"{website}/finish_upload/{upload_id}/", headers=headers)
    if response.status_code != 200:
        print("Failed to finish upload:", response.text)
        exit(1)
    else:
        print("Successfully uploaded your submission. Evaluation will start soon and results will be sent to your email.")


def compress_folder(path):
    tmp_folder = os.path.join(os.path.dirname(path), 'tmp')
    os.makedirs(tmp_folder, exist_ok=True)
    os.system(f"tar -cvzf - {path} | split -a 4 -b 512M -d - {tmp_folder}/submission.tar.gz.")
    return tmp_folder

def main(args):
    tmp_folder = compress_folder(args.path)
    # tmp_folder = os.path.join(os.path.dirname(args.path), 'tmp')
    upload_id = request_verification(args)

    if upload_id:
        code = input("Please enter the verification code sent to your email: ")

        if verify_code(upload_id, code):
            upload_file(upload_id, tmp_folder)
        else:
            print("Verification failed. Upload not allowed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", help="Email address to send verification code to", required=True)
    parser.add_argument("--path", help="Submission path", required=True)
    parser.add_argument("--method_name", help="Method name that will be displayed in the leaderboard", default='RAFT')
    parser.add_argument("--benchmark", help="Benchmark name that will be displayed in the leaderboard", default='first_layer', choices=['first_layer', 'last_layer', 'multi_layer'])
                        
    
    args = parser.parse_args()
    main(args)
