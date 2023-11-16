from crsgpt.agent.agent import *
import logging
import argparse
from crsgpt.evaluator.fake_user import *
from crsgpt.evaluator.pure_gpt import *
from crsgpt.component.product import *
from crsgpt.component.preference import *

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

def main_loop():
    product=Product(args.head_K,file_logger,args.embedding_cache_path,args.top_K,
                    args.update_product,args.verbose,product_dict)
    preference=Preference(file_logger,args.verbose)
    evaluator=Evaluator(file_logger,product.product_type_set,args.verbose)
    rec=Agent(product,preference,evaluator,file_logger,args.explicit,args.verbose,True,args.web)
    gpt_rec=PureGPT(file_logger)
    product_gpt=ProductGPT(product,file_logger,args.explicit,args.verbose)

    tester=GoalTester(file_logger,args.verbose,product)

    user_input=""
    rec_response=""
    context=[]
    times = []
    success = False
    while True:
        user_input=tester.interactive(rec_response)
        context.append({"role":"user","content":user_input})
        print(f'[[User]]: {user_input}')
        if user_input=="success":
            success = True
            break
        elif (len(context)>=15):
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
    
    return context,np.mean(times), success


if __name__=='__main__':
    print(args)
    file_logger = logging.getLogger("file_logger")
    file_logger.setLevel(args.log_level)
    file_hander = logging.FileHandler(args.log_file)
    file_logger.addHandler(file_hander)
    
    api_init()
    
    scores = []
    steps = []
    times = []
    success = []
    



    for i in trange(args.num):
        context,t,s = main_loop()
        scorer=Scorer(file_logger,args.verbose,context,args.explicit)
        score, score_explanation=scorer.score()
        print(f"Score: {score}")
        if args.explicit:
            print(f"Score Explanation: {score_explanation}")
        file_logger.info(f"\n********\nScore: {score}********\n")
        if args.explicit:
            file_logger.info(f"Score Explanation: {score_explanation}\n********\n\n")
        scores.append(score)
        steps.append(len(context))
        times.append(t)
        success.append(s)
    import pickle
    with open("./results/target.pkl", "wb") as f:
        pickle.dump({"scores":scores, "steps":steps, "times":times, "success":success}, f)
    print({"scores":np.mean(scores), "steps":np.mean(steps), "times":np.mean(times), "success":np.mean(success)})
