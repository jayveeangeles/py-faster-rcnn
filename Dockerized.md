## Steps to run dockerized version  

1) Install Docker for your platform. For Windows users, be sure to check if you have to install the [legacy](https://docs.docker.com/toolbox/toolbox_install_windows/) or [current](https://docs.docker.com/docker-for-windows/install/) platform. Check also that virtualization is enabled for your system. 
2) Pull the image from the dockerio registry to run Faster-RCNN by doing a:  
`docker pull vincentj0520/frcnn:powerai-1.6.1-caffe-cpu-ubuntu18.04-py3-x86_64`  
3) Wait for a while (around 2.5 minutes depending on your network connection) :coffee:  
4) Run the inference script from within the container by doing a:  
`docker run --rm -e "LICENSE=yes"  -v $(PWD)/your_model.tgz:/workspace/data/model.tgz -v $(PWD)/your_photo.jpg:/tmp/test.jpg -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=192.168.1.120:0 vincentj0520/frcnn:powerai-1.6.1-caffe-cpu-ubuntu18.04-py3-x86_64`  
5) Wait for a while since the detection takes some time for the CPU only version. Enjoy! 😄  

## References
1) `your_model.tgz` is the tar.gz file containing the model (renamed to model.caffemodel), the deployment prototxt (named test.prototxt) and the labels (in comma separated list enclosed in parenthesis, called classname.txt). Please change the filename accordingly.    
2) `your_photo.jpg` is the picture you want to run inference on. 
3) Do remember to not change the mounting points for the bind mounts. For now, only the image name can be changed by setting the environment variable `IMAGE_NAME` to something other than `/tmp/test.jpg`. If you're not sure, leave the values as is.  
4) On *nix based systems, xhost + must be run to allow the container to display using the host UI. Additionally, for MacOS users. XQuartz needs to be installed. This step is not needed for Linux based systems.  
5) You can change the NMS and confidence thresholds by setting the environment variables `NMS_THRE` and `CONF_THRE` respectively. Override these values when you do step 4.  
6) For more advanced users, the python script to do the inferencing actually has a few arguments that can be changed. Override the default inference process by passing an executable script during the docker run phase.   

## Notes for Windows 10 Users
1) Make sure that your system has virtualization enabled. After installing Docker, reboot your system and when you run Docker, say yes to the option that tells you to give it permission to run virtualization.  
2) Try to give as much memory as possible for the Docker VM in the settings. 4GB should be good enough to run this.  
3) To enable the display, follow this [link](https://dev.to/darksmile92/run-gui-app-in-linux-docker-container-on-windows-host-4kde). The link details how you can output the results in the host display. You will be needing VcXsrv. Install it as per normal using this [link](https://sourceforge.net/projects/vcxsrv/); you won't have to install Chocolatey if you want.  
4) Always take note of your IP address if you're using a non-Linux host. For Windows 10, take note of your DockerNAT IP. For Mac users, try using your en0 interface (still looking for ways to use the bridge interface from Docker).  
5) For Windows mounting, follow *nix based directory paths for your home folder. For example, if your directory is in `C:\Users/vincentj\Documents\test\out.tgz`, mount it by using the path `c:/Users/vincentj/Documents/test/out.tgz` so that Docker recognizes your mount point.  
6) Try to use PowerShell if possible.  

## To Do's
1) Update the command to run for Windows-based Users. The display variables are for *nix based systems (verified with MacOS; for Linux, the IP isn't needed).  
