from crsgpt.agent.agent import *
import logging
import argparse
from crsgpt.evaluator.fake_user import *
from crsgpt.evaluator.pure_gpt import *
from crsgpt.component.product import *
from crsgpt.component.preference import *
from tqdm import trange

parser = argparse.ArgumentParser(description = 'test')

parser.add_argument('--head_K', type=int, default=50, help='head K products in system')
parser.add_argument('--top_K', type=int, default=5, help='top K products to recommend')
parser.add_argument('--num', type=int, default=10, help='top K products to recommend')
parser.add_argument('--log_file', type=str, default="./logs/func.log",help='log file path')
parser.add_argument('--update_product', action='store_true', default=False,help='whether to update product details')
parser.add_argument('--embedding_cache_path', type=str, default="./data/recommendations_embeddings_cache.pkl",help='product details json save path')
parser.add_argument('--log_level', type=int, default=logging.DEBUG,help='log level')
parser.add_argument('--explicit', action='store_true',help='whether to explicitly show the thinking of the recommendation gpt')
parser.add_argument('--verbose', action='store_true',help='whether to show the process of the recommendation')
parser.add_argument('--testcase', nargs='*',  default=None,help='testcase')
parser.add_argument('--product_gpt', action='store_true')
parser.add_argument('--pure_gpt', action='store_true')
parser.add_argument('--web', action='store_true')

args = parser.parse_args()

product_dict = {
    './data/2022_movie.csv':'Title',
    './data/2022_book.csv':'Name_of_the_Book',
    './data/2023_phone.csv':'name',
    './data/2023_movie.csv':'name'
}

def main_loop(testcase=None):
    product=Product(args.head_K,file_logger,args.embedding_cache_path,args.top_K,
                    args.update_product,args.verbose,product_dict)
    preference=Preference(file_logger,args.verbose)
    evaluator=Evaluator(file_logger,product.product_type_set,args.verbose)
    rec=Agent(product,preference,evaluator,file_logger,args.explicit,args.verbose,True,args.web)
    gpt_rec=PureGPT(file_logger)
    product_gpt=ProductGPT(product,file_logger,args.explicit,args.verbose)

    if testcase is not None:
        tester=Tester(file_logger,args.verbose,product.product_type_set,testcase)

    user_input=""
    rec_response=""
    context=[]
    times = []
    while True:
        if testcase is not None:
            user_input=tester.interactive(rec_response)
        else:
            user_input=input("User: ")
        context.append({"role":"user","content":user_input})
        print(f'[[User]]: {user_input}')
        if user_input=="exit" or ((testcase is not None) and len(context)>=10):
            break
        rec_response,t=rec.user_interactive(user_input)
        times.append(t)
        print(f'[[Rec]]: {rec_response}')
        if args.product_gpt:
            product_gpt_response=product_gpt.user_interactive(context)
            print(f'[[ProductGPT]]: {product_gpt_response}')
        if args.pure_gpt:
            gpt_rec_response=gpt_rec.interactive(context)
            print(f'[[ChatGPT]]: {gpt_rec_response}')
        context.append({"role":"assistant","content":rec_response})
    
    return context,np.mean(times)


if __name__=='__main__':
    print(args)
    file_logger = logging.getLogger("file_logger")
    file_logger.setLevel(args.log_level)
    file_hander = logging.FileHandler(args.log_file)
    file_logger.addHandler(file_hander)
    
    api_init()
    
    scores = {}
    times = {}


    testcases = args.testcase if args.testcase[0] != "all" else TestPrompter.TEST_CASES.keys()
    for testcase in testcases:
        print(f"*****Testcase: {testcase}*****")
        scores[testcase] = []
        times[testcase] = []
        file_logger.info(f"\n\n*****Testcase: {testcase}*****")
        for i in trange(args.num):
            context,t = main_loop(testcase)
            scorer=Scorer(file_logger,args.verbose,context,args.explicit)
            score, score_explanation=scorer.score(testcase)
            print(f"Score: {score}")
            if args.explicit:
                print(f"Score Explanation: {score_explanation}")
            file_logger.info(f"\n********\nScore: {score}********\n")
            if args.explicit:
                file_logger.info(f"Score Explanation: {score_explanation}\n********\n\n")
            scores[testcase].append(score)
            times[testcase].append(t)
    import pickle
    with open("./results/function.pkl", "wb") as f:
        pickle.dump({"scores":scores, "times":times}, f)
