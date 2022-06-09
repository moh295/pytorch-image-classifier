//
// Created by mom44 on 09/06/2022.
//
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include "printfiles2.h"



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
