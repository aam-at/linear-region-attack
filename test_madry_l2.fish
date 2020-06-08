for i in (seq 0 499)
    python main_madry_mnist.py mnist_madry_l2 --working_dir="results_mnist_l2/image_$i" --image $i
end
