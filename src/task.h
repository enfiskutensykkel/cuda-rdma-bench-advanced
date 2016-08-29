#ifndef __TASK_H__
#define __TASK_H__

#include <string>


struct Task
{
    std::string     remoteHostName;
    unsigned int    localAdapterNo;
    unsigned int    remoteNodeId;
};

#endif
