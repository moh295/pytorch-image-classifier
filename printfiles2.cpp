#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
using namespace std;

using std::cout; using std::cin;
using std::endl; using std::vector;
//using std::filesystem::directory_iterator;




int find_file_name_list() {
    DIR *dir; struct dirent *diread;
    vector<char *> files;

    if ((dir = opendir("/")) != nullptr) {
        while ((diread = readdir(dir)) != nullptr) {
            files.push_back(diread->d_name);
        }
        closedir (dir);
    } else {
        perror ("opendir");
        return EXIT_FAILURE;
    }

    for (auto file : files) cout << file << "| ";
    cout << endl;

    return EXIT_SUCCESS;
}



int main() {
    std::cout << "Hello, World!" << std::endl;




    find_file_name_list();
//    for (int i=0;i<20;i++)
//        std::cout << "i ="<< i <<std::endl;
    return 0;
}
