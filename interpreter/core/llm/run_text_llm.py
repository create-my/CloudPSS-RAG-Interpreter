def run_text_llm(llm, params):
    ## Setup

    if llm.execution_instructions:
        try:
            # Add the system message
            params["messages"][0][
                "content"
            ] += "\n" + llm.execution_instructions
        except:
            print('params["messages"][0]', params["messages"][0])
            raise

    ## Convert output to LMC format

    inside_code_block = False
    accumulated_block = ""
    language = None
    if llm.completions.__name__ == 'cloudpss_agent':#袁雪峰修改，如果调用dcloudpssagent
        params["conversation_id"] = llm.coversation_id
    for chunk in llm.completions(**params):
        if llm.interpreter.verbose:
            print("Chunk in coding_llm", chunk)

        if "choices" not in chunk or len(chunk["choices"]) == 0:
            # This happens sometimes
            continue

        content = chunk["choices"][0]["delta"].get("content", "")

        if content == None:
            continue

        accumulated_block += content

        if accumulated_block.endswith("`"):
            # We might be writing "```" one token at a time.
            # 如果content以`结尾，前面可能是代码，应当将代码部分发送出去
            if language:
                yield {
                    "type": "code",
                    "format": language,
                    "content": content.split("`")[0], # 袁雪峰添加yield
                }
            continue

        # Did we just enter a code block?
        if "```" in accumulated_block and not inside_code_block:
            inside_code_block = True
            accumulated_block = accumulated_block.split("```")[1]

        # Did we just exit a code block?
        if inside_code_block and "```" in accumulated_block:
            # print('代码块结束了', accumulated_block)

            # if content.split("`")[0]== '':
            #     finally_code = content.split("`")[0]
            # else:
            #     finally_code = content.split("\n")[0] #袁雪峰添加，最后一段有可能是代码，需要判断
            # yield {
            #     "type": "code",
            #     "format": language,
            #     "content": finally_code,  # 这里直接打印出去是有问题的。content此时如果是n\ndef，
            # }
            return

        # If we're in a code block,
        if inside_code_block:

            # If we don't have a `language`, find it
            if language is None and "\n" in accumulated_block:
                language = accumulated_block.split("\n")[0]
                content = accumulated_block # 袁雪峰添加，只有第一次判断language才会进来，后面都是代码
                # Default to python if not specified
                if language == "":
                    if llm.interpreter.os == False:
                        language = "python"
                    elif llm.interpreter.os == False:
                        # OS mode does this frequently. Takes notes with markdown code blocks
                        language = "text"
                else:
                    # Removes hallucinations containing spaces or non letters.
                    language = "".join(char for char in language if char.isalpha())

            # If we do have a `language`, send it out
            #这里判断language，应该第一次不判断，第二次才判断。或者第一次修改content内容
            if language:
                yield {
                    "type": "code",
                    "format": language,
                    "content": content.replace(language, ""),# 这里直接打印出去是有问题的。content此时如果是n\ndef，
                }

        # If we're not in a code block, send the output as a message
        if not inside_code_block:
            yield {"type": "message", "content": content}