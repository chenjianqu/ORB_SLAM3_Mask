/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <filesystem>

#include<opencv2/core/core.hpp>
#include <torch/torch.h>

#include<System.h>

using namespace std;

namespace fs = std::filesystem;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

torch::Tensor LoadTensor(const string &load_path);
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> LoadMaskTensor(string seq_id,const fs::path &mask_dir);
cv::Mat VisualTensor(torch::Tensor &seg_label);

cv::Mat GetMask(const string &file_name,const fs::path &mask_dir);


int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
    cv::Mat imMask;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);

        if(SLAM.useMask()){
            fs::path rgb_path(vstrImageFilenamesRGB[ni]);
            imMask = GetMask(rgb_path.stem(),argv[3]);

            cv::imshow("mask",imMask);
            cv::waitKey(1);
        }

        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        if(!SLAM.useMask()){
            // Pass the image to the SLAM system
            SLAM.TrackRGBD(imRGB,imD,tframe);
        }
        else{
            SLAM.TrackRGBD(imRGB,imD,tframe,vector<ORB_SLAM3::IMU::Point>(),"",imMask);
        }


#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");   

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}




torch::Tensor LoadTensor(const string &load_path){

    cout<<load_path<<endl;

    std::ifstream input(load_path, std::ios::binary);
    std::vector<char> bytes( (std::istreambuf_iterator<char>(input)),
                             (std::istreambuf_iterator<char>()));
    input.close();

    torch::IValue x = torch::pickle_load(bytes);
    torch::Tensor tensor = x.toTensor();
    return tensor;
}


std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> LoadMaskTensor(string seq_id,const fs::path &mask_dir){
    fs::path seg_label_path = mask_dir / ("seg_label_"+ seq_id+".pt");
    if(!fs::exists(seg_label_path)){
        cerr<<seg_label_path<<" does not exist"<<endl;
        return {};
    }

    torch::Tensor seg_label = LoadTensor(mask_dir / ("seg_label_"+ seq_id+".pt"));
    torch::Tensor cate_score = LoadTensor(mask_dir / ("cate_score_"+ seq_id+".pt"));
    torch::Tensor cate_label = LoadTensor(mask_dir / ("cate_label_"+ seq_id+".pt"));
    return {seg_label,cate_score,cate_label};
}


cv::Mat VisualTensor(torch::Tensor &seg_label)
{
    int cols = seg_label.size(2);
    int rows = seg_label.size(1);
    int num_mask = seg_label.size(0);

    auto mask_size=cv::Size(cols ,rows);
    auto mask_tensor = seg_label.to(torch::kInt8).abs().clamp(0,1);

    ///计算合并的mask
    auto merge_tensor = (mask_tensor.sum(0).clamp(0,1)*255).to(torch::kUInt8).to(torch::kCPU);
    auto mask = cv::Mat(mask_size,CV_8UC1,merge_tensor.data_ptr()).clone();

    //cv::cvtColor(mask,mask,CV_GRAY2BGR);
    //auto color = getRandomColor();

    return mask;
}


cv::Mat GetMask(const string &file_name,const fs::path &mask_dir){

    auto [seg_label,cate_score,cate_label] = LoadMaskTensor(file_name,mask_dir/"mask");
    if(seg_label.defined()){
        ///只保留人的类别
        torch::Tensor vis_inds = cate_label == 0;
        //vis_inds += (cate_label == 1);
        vis_inds=vis_inds.toType(torch::kLong);
        auto index = vis_inds.nonzero().squeeze();

        cate_label = cate_label.index_select(0,index);
        cate_score = cate_score.index_select(0,index);
        seg_label = seg_label.index_select(0,index);

        return VisualTensor(seg_label);
    }
    else{
        return {};
    }
}


