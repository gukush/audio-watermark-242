import os
import subprocess
import gdown
import signal

def download_audio(url,output):
    if not os.path.isfile(output):
        gdown.download(url=url,output=output,fuzzy=True)
if not os.path.exists('./audio/old'):
    os.makedirs('./audio/old')
print("Getting test audio samples")
root_path = os.path.dirname(__file__)
download_audio(
    'https://drive.google.com/file/d/1FoeK7httIPp5qJ9PlZifDzNimTYzv_Rl/view?usp=drive_link',#'https://drive.google.com/file/d/1QTGpn2V55tlh_WI9JFU-IMd7DAnqfF_y/view?usp=drive_link',
    os.path.join(root_path,'audio','old','voice-polish-1.wav')
)
download_audio(
    'https://drive.google.com/file/d/1nGCLgDkUiPbdaocO78xh-toYzWkEHcDQ/view?usp=drive_link',#'https://drive.google.com/file/d/1nu5V-IYkLEI6ZjQaiBLTLfarpgj7h-nh/view?usp=drive_link',
    os.path.join(root_path,'audio','old','voice-hispanic-1.wav')
)
download_audio(
    'https://drive.google.com/file/d/1p1DHko7lbus8DKsVzCj830jHQCWOgtMI/view?usp=drive_link',#'https://drive.google.com/file/d/12aVcw-Ca4i5qq1nWb6LmcXWajjjzGzTk/view?usp=drive_link',
    os.path.join(root_path,'audio','old','voice-polish-8.wav')
)

dir_path = './tests'
passed_tests = []
n = 0
for test_file in os.listdir('./tests'):
    full_path = os.path.join(dir_path,test_file)
    if os.path.isfile(full_path) and os.path.getsize(full_path) != 0:
        print("------------------------")
        print(f"Running test: {test_file}")
        print("------------------------")
        n = n + 1 # quick maths
        try:
            result = subprocess.run(
                ['python',full_path],
                text=True,
                capture_output=True,
                check=True
            )
            if result.returncode < 0:
                signal_num = -result.returncode
                if signal_num == signal.SIGKILL:
                    print(f"Test failed: {test_file} - killed")
                else:
                    print(f"Test had bad returncode: {test_file}")
            else:
                passed_tests.append(test_file)
                print(f"Test passed: {test_file}")
            print("------------------------")
            print(result.stdout)
            print("------------------------")
        except subprocess.CalledProcessError as e:
            print("------------------------")
            print("Script crashed with return code:", e.returncode)
            print("Error output:", e.stderr)
            print("------------------------")
        except Exception as general_error:
            print("An unexpected error occurred:", str(general_error))
            print("------------------------")
print(f"{len(passed_tests)}/{n} tests passed")
