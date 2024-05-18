import multiprocessing

def run_script1():
    import test1  
    test1.main()

def run_script2():
    import test2  
    test2.main()

if __name__ == "__main__":
    process1 = multiprocessing.Process(target=run_script1)
    process2 = multiprocessing.Process(target=run_script2)

    process1.start()
    process2.start()

    process1.join()
    process2.join()
