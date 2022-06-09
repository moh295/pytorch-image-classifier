#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
using namespace std;

using std::cout; using std::cin;
using std::endl; using std::vector;
//using std::filesystem::directory_iterator;

bool find_sub_string(string, string);
vector<string>  find_file_name_list();


vector<string> find_file_name_list() {
    DIR *dir; struct dirent *diread;
    vector<char *> files;

    if ((dir = opendir("/media/workspace/simple-classifier/data/sampleFasterRCNN/faster-rcnn")) != nullptr) {
        while ((diread = readdir(dir)) != nullptr) {


            if (find_sub_string(diread->d_name,".ppm"))
            files.push_back(diread->d_name);
        }
        closedir (dir);
    } else {
        perror ("no such folder");
        //return 'e';
    }

    for (auto file : files) cout << file << "| ";
    cout << endl;

    return files;
}



bool find_sub_string(std::string s1 ,std::string s2){

    if (s1.std::string::find(s2) != std::string::npos) {
        std::cout << "found ppm file!" << '\n';

        return true;
    }
    else return false;
}



int main() {
    std::cout << "Hello, World!" << std::endl;

    vector<string> files;


    files=find_file_name_list();
    for (auto f: files) cout<< "filename:" << f <<'\n';

// cout<< "filename:" << files <<'\n';
//    for (int i=0;i<20;i++)
//        std::cout << "i ="<< i <<std::endl;
    return 0;
}
