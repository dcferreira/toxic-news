# import os
#
# import pytest
# from xprocess import ProcessStarter
#
#
# class TestDockerServer:
#     @pytest.fixture(scope="module")
#     def setup(self):
#         os.system("hatch run build")
#
#     @pytest.fixture(scope="module")
#     def server_docker(self, xprocess, setup):
#         class Starter(ProcessStarter):
#             pattern = "Application startup complete"
#             args = ["hatch", "run", "serve-docker"]
#
#         xprocess.ensure("server-docker", Starter)
#         url = "http://127.0.0.1:5000"
#         yield url
#
#         xprocess.getinfo("server-docker").terminate()
