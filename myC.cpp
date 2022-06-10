#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>

#include <chrono>
#include <thread>
using namespace std;

using std::cout; using std::cin;
using std::endl; using std::vector;
//using std::filesystem::directory_iterator;

bool find_sub_string(string, string);
vector<string>  find_file_name_list(string);
float elapsed(std::chrono::system_clock::time_point );

vector<string> find_file_name_list(string imgefolder) {
    DIR *dir; struct dirent *diread;
    vector<string> files;

    if ((dir = opendir(imgefolder)) != nullptr) {
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
float elapsed(std::chrono::system_clock::time_point time_then){

    auto timeInMicroSec=std::chrono::high_resolution_clock::now()-time_then;
    float elapsed(timeInMicroSec.count());

    cout << "elapsed:"<< elapsed/1000000000;
    return elapsed;

}

int main() {
    std::cout << "Hello, World!" << std::endl;

    vector<string> files;

    auto then = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for( std::chrono::seconds(3) );
    elapsed(then);


    //files=find_file_name_list();
    //for (auto f: files) cout<< "filename:" << f <<'\n';


//    for (int i=0;i<20;i++)
//        std::cout << "i ="<< i <<std::endl;
    return 0;
}
