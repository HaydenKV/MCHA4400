#include <cstdlib>
#include <cstdio>
#include <string>
#include <sstream>
#include <print>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <nanobench.h>
#include "imagefeatures.h"

int main(int argc, char *argv[])
{
    cv::String keys = 
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this message}"
        "{@input          | <none>   | input can be a path to an image or video (e.g., ../data/lab.jpg)}"
        "{export e        |          | export output file to the ./out/ directory}"
        "{N               | 10       | maximum number of features to find}"
        "{detector d      | fast     | feature detector to use (e.g., harris, shi, aruco, fast)}"
        "{benchmark b     |          | run benchmark for all detectors}"
    ;
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Lab 2");

    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    // Parse input arguments
    bool doExport = parser.has("export");
    int maxNumFeatures = parser.get<int>("N");
    bool doBenchmark = parser.has("benchmark");
    cv::String detector = parser.get<std::string>("detector");
    std::filesystem::path inputPath = parser.get<std::string>("@input");

    // Check for syntax errors
    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return EXIT_FAILURE;
    }

    if (!std::filesystem::exists(inputPath))
    {
        std::println("File: {} does not exist", inputPath.string());
        return EXIT_FAILURE;
    }

    // Prepare output directory
    std::filesystem::path outputDirectory;
    if (doExport)
    {
        std::filesystem::path appPath = parser.getPathToApplication();
        outputDirectory = appPath / ".." / "out";

        // Create output directory if we need to
        if (!std::filesystem::exists(outputDirectory))
        {
            std::println("Creating directory {}", outputDirectory.string());
            std::filesystem::create_directory(outputDirectory);
        }
        std::println("Output directory set to {}", outputDirectory.string());
    }

    // Prepare output file path
    std::filesystem::path outputPath;
    if (doExport)
    {
        std::string outputFilename = inputPath.stem().string()
                                   + "_"
                                   + detector
                                   + inputPath.extension().string();
        outputPath = outputDirectory / outputFilename;
        std::println("Output name: {}", outputPath.string());
    }

    // Check if input is an image or video (or neither)
    bool isVideo = false; // TODO
    bool isImage = false; // TODO

    if (!isImage && !isVideo)
    {
        std::println("Could not read file: {}", inputPath.string());
        return EXIT_FAILURE;
    }

    if (doBenchmark)
    {
        if (!isImage)
        {
            std::println("Benchmark can only be run on images, not videos.");
            return EXIT_FAILURE;
        }

        // Suppress console output during benchmark
        std::FILE* old_stdout = stdout;
        stdout = std::fopen("/dev/null", "w");

        // Create benchmark object
        ankerl::nanobench::Bench bench;
        
        // Capture benchmark output in a separate stringstream
        std::stringstream bench_output;
        bench.output(&bench_output);

        // TODO: Run the benchmarks for the 4 feature detectors


        // Restore console output
        std::fclose(stdout);
        stdout = old_stdout;

        // Print benchmark results
        std::println("\nBenchmark results:");
        std::print("{}", bench_output.str());

        return EXIT_SUCCESS;
    }

    if (isImage)
    {
        // TODO: Call one of the detectAndDraw functions from imagefeatures.cpp according to the detector option specified at the command line
        
        if (doExport)
        {
            // TODO: Write image returned from detectAndDraw to outputPath
        }
        else
        {
            // TODO: Display image returned from detectAndDraw on screen and wait for keypress
        }
    }

    if (isVideo)
    {
        if (doExport)
        {
            // TODO: Open output video for writing using the same fps as the input video
            //       and the codec set to cv::VideoWriter::fourcc('m', 'p', '4', 'v')
        }

        while (true)
        {
            // TODO: Get next frame from input video

            // TODO: If frame is empty, break out of the while loop
            
            // TODO: Call one of the detectAndDraw functions from imagefeatures.cpp according to the detector option specified at the command line

            if (doExport)
            {
                // TODO: Write image returned from detectAndDraw to frame of output video
            }
            else
            {
                // TODO: Display image returned from detectAndDraw on screen and wait for 1000/fps milliseconds
            }
        }

        // TODO: release the input video object

        if (doExport)
        {
            // TODO: release the output video object
        }
    }

    return EXIT_SUCCESS;
}



