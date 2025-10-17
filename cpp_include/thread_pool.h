#include <thread>
#include <queue>
#include <atomic>
#ifdef _MSC_VER
#include <BaseTsd.h>
using ssize_t = SSIZE_T;
#endif
#include <mutex>

class ThreadPool
{
public:
    // Constructor: Creates a pool of 'num_threads' worker threads
    ThreadPool() : ThreadPool(std::thread::hardware_concurrency())
    {
    }
    ThreadPool(ssize_t num_threads) : stop_all(false), active_tasks(0)
    {
        for (ssize_t i = 0; i < num_threads; ++i)
        {
            // Create and detach threads, each running the 'worker_loop'
            workers.emplace_back([this]
                                 {
                // Each worker runs an infinite loop waiting for tasks
                while (true) {
                    std::function<void()> task;

                    // Acquire lock to safely access the task queue
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        
                        // Wait until the queue is not empty OR the pool is stopping
                        cv.wait(lock, [this]{
                            return stop_all || !tasks.empty();
                        });

                        // If the pool is stopping AND the queue is empty, exit the thread
                        if (stop_all && tasks.empty()) {
                            return;
                        }

                        // Get the next task and remove it from the queue
                        task = std::move(tasks.front());
                        tasks.pop();
                    } // Lock is automatically released here

                    // Execute the task
                    task();

                    // Mark task completion and notify waiters if no more work is pending
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        active_tasks--;
                        if (tasks.empty() && active_tasks.load() == 0)
                        {
                            done_cv.notify_all();
                        }
                    }
                } });
        }
    }

    // Destructor: Stops all worker threads
    ~ThreadPool()
    {
        // Signal to all threads to stop and wake up any waiting threads
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop_all = true;
        }
        cv.notify_all();

        // Join all worker threads to ensure they finish execution
        for (std::thread &worker : workers)
        {
            worker.join();
        }
    }

    // Enqueues a new task (a function object) to the pool
    void enqueue(std::function<void()> task)
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            active_tasks++;
            tasks.emplace(std::move(task)); // Add the task to the queue
        }
        cv.notify_one(); // Wake up one waiting worker thread
    }
    void wait_all()
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        done_cv.wait(lock, [this]()
                     { return tasks.empty() && active_tasks.load() == 0; });
    }

private:
    std::vector<std::thread> workers;        // The thread pool
    std::queue<std::function<void()>> tasks; // The thread-safe task queue

    std::mutex queue_mutex;           // Mutex for synchronizing access to the queue
    std::condition_variable cv;       // Condition variable to wait/notify on new tasks
    bool stop_all;                    // Flag to indicate when threads should stop
    std::atomic<size_t> active_tasks; // Count of tasks enqueued but not yet completed
    std::condition_variable done_cv;  // Notifies when all tasks are completed
};
