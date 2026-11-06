
#include <iostream>
#include <string>
#include "openvino/genai/llm_pipeline.hpp"

int main(int argc,char* argv[]){
    if(argc<2){
        std::cerr<<"Usage: "<<argv[0]<<" <model_dir>\n";
        return 1;
    }

    std::string model_path=argv[1];

    ov::genai::LLMPipeline pipe(model_path,"CPU");

    ov::genai::GenerationConfig config;
    config.max_new_tokens=100;

    auto streamer=[](std::string token)->ov::genai::StreamingStatus{
        std::cout<<token<<std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };

    std::string prompt;

    std::cout<<"TinyLlama Local Chat\n";
    std::cout<<"Type 'exit' to quit.\n\n";

    while(true){
        std::cout<<"You: ";
        std::getline(std::cin,prompt);

        if(prompt=="exit"){
            std::cout<<"Chat ended.\n";
            break;
        }

        if(prompt.empty()){
            continue;
        }

        std::cout<<"TinyLlama: ";

        auto result=pipe.generate(prompt,config,streamer);

        std::cout<<"\n\n";
    }

    return 0;
}
