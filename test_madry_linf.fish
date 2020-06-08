for i in (seq 0 499)
    python main_madry_mnist.py mnist_madry_linf --working_dir="results_mnist_linf/image_$i" --image $i
end
