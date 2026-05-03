### Task adaptation

To adapt PlugMem to HotpotQA with deeper multi-hop reasoning and a stronger backbone, only two parameters in `src/eval/hotpotqa/eval_qa_all.py` need to be changed:

- `--n_round_retrieval` from the default `2` to `3` (allows one additional retrieval round)
- `--qa_model_name` to `gpt-5.4`

Then export your OpenAI credentials before running the script:

```bash
export OPENAI_BASE_URL="<your endpoint>"
export OPENAI_API_KEY="<your key>"
```

### Performance
To further assess the potential of PlugMem as a memory backbone, we increase the maximum multi-hop depth $T_{\max}$ from 2 to 3 and adopt a stronger backbone model, GPT-5.4. Under this setting, PlugMem achieves an F1 score of **79.1**, substantially outperforming the **74.1** F1 score reported in the main experiments. In addition to token-level F1, we further evaluate answer correctness using LLM-as-Judge accuracy, where this configuration achieves **91.1%**.

To estimate the upper bound under ideal evidence retrieval, we report an oracle baseline that directly answers with the gold paragraphs as context, which obtains an F1 score of **83.8** and an LLM-as-Judge accuracy of **95.0%**. After task adaptation with deeper reasoning depth and a stronger backbone model, PlugMem reaches **94.4%** of the oracle F1 score and **95.9%** of the oracle LLM-as-Judge accuracy.
