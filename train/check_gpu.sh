./gpu_burn -tc 10 | grep FAULTY > /dev/null
if [ $? -eq 0 ]; then
    hostname
fi
