//
// Created by mom44 on 09/06/2022.
//
#include <iostream>
#include <string>
#include <filesystem>

using std::cout; using std::cin;
using std::endl; using std::string;
using std::filesystem::directory_iterator;


int main() {
    std::cout << "Hello, World!" << std::endl;



    string path = "C:/Users/mom44/Desktop/temp ppm images";

    for (const auto & file : directory_iterator(path))
        cout << file.path() << endl;

    return EXIT_SUCCESS;

    return 0;
}

