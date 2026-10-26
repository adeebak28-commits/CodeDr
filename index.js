import { GoogleGenAI } from "@google/genai";
import 'dotenv/config'
import fs from 'fs';
import path from 'path';

if (!process.env.GEMINI_API) {
    console.error("Error: GEMINI_API is missing. Add it to your .env file.");
    process.exit(1);
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API });

const MAX_ITERATIONS = 50;   // safety net so the agent loop can never run forever
const MAX_RETRIES = 3;       // retries for temporary API failures

// HELPERS

function errorMessage(err)
{
    return err instanceof Error ? err.message : String(err);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// TOOL FUNCTIONS

async function listFiles({ directory })
{
    try
    {
        if (!directory || !fs.existsSync(directory))
            return { error: `Directory not found: ${directory}` };

        if (!fs.statSync(directory).isDirectory())
            return { error: `Not a directory: ${directory}` };

        const files = [];
        const skipped = [];
        const extensions = ['.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.json'];

        function scan(dir)
        {
            let items;
            try
            {
                items = fs.readdirSync(dir);
            }
            catch (err)
            {
                // unreadable folder (permissions etc.) - skip it, keep scanning the rest
                skipped.push(dir);
                console.warn(`Skipping ${dir}: ${errorMessage(err)}`);
                return;
            }

            for (const item of items)
            {
                const fullPath = path.join(dir, item);

                //skip node_modules , dist , build
                if (fullPath.includes('node_modules') || fullPath.includes('dist') || fullPath.includes('build'))
                    continue;

                try
                {
                    const stat = fs.statSync(fullPath);

                    if (stat.isDirectory())
                        scan(fullPath);
                    else if (stat.isFile())
                    {
                        const ext = path.extname(item);
                        if (extensions.includes(ext))
                            files.push(fullPath);
                    }
                }
                catch (err)
                {
                    // broken symlink, permission error, etc.
                    skipped.push(fullPath);
                    console.warn(`Skipping ${fullPath}: ${errorMessage(err)}`);
                }
            }
        }

        scan(directory);
        console.log(`Found ${files.length} files`);
        return skipped.length > 0 ? { files, skipped } : { files };
    }
    catch (err)
    {
        console.error(`list_files failed: ${errorMessage(err)}`);
        return { error: `Failed to list files: ${errorMessage(err)}` };
    }
}

async function readFile({ file_path })
{
    try
    {
        const content = fs.readFileSync(file_path, 'utf-8');
        console.log(`Reading : ${file_path}`);
        return { content };
    }
    catch (err)
    {
        console.error(`read_file failed for ${file_path}: ${errorMessage(err)}`);
        return { error: `Failed to read ${file_path}: ${errorMessage(err)}` };
    }
}

async function writeFile({ file_path, content })
{
    try
    {
        if (typeof content !== 'string')
            return { success: false, error: "content must be a string" };

        fs.writeFileSync(file_path, content, 'utf-8');
        console.log(`Fixed : ${file_path}`);
        return { success: true };
    }
    catch (err)
    {
        console.error(`write_file failed for ${file_path}: ${errorMessage(err)}`);
        return { success: false, error: `Failed to write ${file_path}: ${errorMessage(err)}` };
    }
}

// TOOLS DECLARATIONS

const listFilesTool = {
    name : "list_files",
    description : "Get all Javascript files in a directory.",
    parameters : {
        type : "OBJECT",
        properties: {
            directory : {
                type : "STRING",
                description : "Directory path to scan."
            }
        },
        required : ["directory"]
    },
};

const readFileTool = {
    name : "read_file",
    description : "Read a file's content.",
    parameters : {
        type : "OBJECT",
        properties: {
            file_path : {
                type : "STRING",
                description : "Path to the file."
            }
        },
        required : ["file_path"]
    },

};

const writeFileTool = {
    name : "write_file",
    description : "Write fixed content back to a file.",
    parameters : {
        type : "OBJECT",
        properties: {
            file_path : {
                type : "STRING",
                description : "Path to the file to write."
            },
            content : {
                type : "STRING",
                description : "The fixed/corrected content."
            }
        },
        required : ["file_path" , "content"]
    },
}

// TOOL REGISTRY

const toolDeclarations = {
    functionDeclarations: [
        listFilesTool,
        readFileTool,
        writeFileTool
    ]
};

const toolHandlers = {
    list_files: listFiles,
    read_file: readFile,
    write_file: writeFile
};

// SYSTEM INSTRUCTION

const SYSTEM_INSTRUCTION = "You are an expert Javascript code reviewer and fixer. Your job : 1) Use list_files to get all HTML , CSS , javascript and typescript files in the directory. 2) Use read_file to read each file's content 3) Analyze for : **HTML Issues** - missing doctype,meta tags,semantic html - broken links,missing alt attributes - accessibility issues(aria,roles) - inline styles that should be in CSS. **CSS issues** - syntax errors ,invalid properties - browser compatibility issues - insufficient selectors -missing vendor properties - umused or duplicate styles. ** Javascript Issues** - BUGS : null/undefined errors,missing returns , type issues , async issues, async problems - SECURITY : hardcoded secrets, eval(), xss risks, injection vulnerabilities - CODE QUALITY : console.logs, unused code, bad naming , complex logic. 4) Use write_file to FIX the issues you found (write corrected code) 5. After fixing all files , respond with a summary report in the text format. **SUMMARY REPORT FORMAT** CODE REVIWE COMPLETE Total files analyzed : x , files fixed : y . SECURITY FIXES : -files.js:line - fixed hardcoded API key - auth.js:line-removed eval() usage. **BUG FIXES** app.js:line:- added null check for the user object - index.html:line - added missing alt attribute. **CODE QUALITY IMPROVEMENTS** - styles.css:file - removed duplicate styles - script.js:line : removed console.log statements. Be practical and focus on real issues . Actually fix the code , don't just report. ";

// API CALL WITH RETRY

async function generateWithRetry(history)
{
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++)
    {
        try
        {
            return await ai.models.generateContent({
                model : "gemini-3.5-flash",
                contents : history,
                config : {
                    systemInstruction : SYSTEM_INSTRUCTION,
                    tools : [toolDeclarations]
                }
            });
        }
        catch (err)
        {
            // retry only temporary problems: rate limit (429) and server errors (5xx)
            const status = err?.status ?? err?.code;
            const retryable = status === 429 || (typeof status === 'number' && status >= 500);

            if (!retryable || attempt === MAX_RETRIES)
                throw new Error(`Gemini API request failed: ${errorMessage(err)}`);

            const delay = 1000 * 2 ** (attempt - 1);   // 1s, 2s, 4s...
            console.warn(`API error (${status}). Retrying in ${delay / 1000}s... (${attempt}/${MAX_RETRIES})`);
            await sleep(delay);
        }
    }
}

// MAIN FUNCTION

export async function runAgent(directoryPath)
{
    if (!directoryPath || !fs.existsSync(directoryPath))
        throw new Error(`Directory does not exist: ${directoryPath}`);

    console.log(`Reviewing : ${directoryPath} \n`);

    const History = [{
        role : 'user',
        parts : [{ text : `Review and fix all Javascript code in : ${directoryPath}` }]
    }];

    for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++)
    {
        const result = await generateWithRetry(History);

        // process all functions calls at once
        if (result.functionCalls?.length > 0)
        {
            const modelContent = result.candidates?.[0]?.content;

            if (!modelContent)
                throw new Error("Model returned function calls but no content.");

            // Preserve the original model response and its metadata
            History.push(modelContent);

            const responseParts = [];

            for (const functionCall of result.functionCalls)
            {
                const { name, args, id } = functionCall;

                console.log(`Executing: ${name}`);

                const handler = toolHandlers[name];
                let toolResponse;

                if (!handler)
                {
                    // report back to the model instead of crashing the whole run
                    toolResponse = { error: `Unknown tool: ${name}` };
                }
                else
                {
                    try
                    {
                        toolResponse = await handler(args ?? {});
                    }
                    catch (err)
                    {
                        console.error(`Tool ${name} crashed: ${errorMessage(err)}`);
                        toolResponse = { error: `Tool ${name} failed: ${errorMessage(err)}` };
                    }
                }

                responseParts.push({
                    functionResponse: {
                        name,
                        id,
                        response: {
                            result: toolResponse
                        }
                    }
                });
            }

            // Send all function responses back together
            History.push({
                role: "user",
                parts: responseParts
            });
        }
        else
        {
            console.log("\n" + (result.text ?? "(Model returned no text.)"));
            return;
        }
    }

    throw new Error(`Stopped after ${MAX_ITERATIONS} iterations without a final response.`);
}

// ENTRY POINT

const directory = process.argv[2] || '.';

try
{
    await runAgent(directory);
}
catch (err)
{
    console.error(`\nAgent failed: ${errorMessage(err)}`);
    process.exit(1);
}