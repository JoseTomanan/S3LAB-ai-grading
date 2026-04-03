<script lang="ts">
  import * as Sheet from '$lib/components/ui/sheet/index.ts';
  import { marked } from 'marked';
  // @ts-ignore
  import manualRaw from './Manual.md';

  let manual = marked.parse(manualRaw);
</script>



<Sheet.Content side="bottom"
                class="flex max-h-[80dvh] flex-col px-4 py-4">
  <h1>User manual</h1>
  <span class="-mt-6"></span>

  <div class="overflow-y-auto text-sm border-t border-b border-border
              space-y-4 pt-4 pr-2 -mr-2">
    <p class="text-muted-foreground">
      Welcome to <strong>SIPAT.MATH</strong> — an AI-powered grading tool for scanned student
      assessments.
      To serve as a guide for testers, the following expected user flow is devised.
    </p>
    
    <div id="markdown-container">{@html manual}</div>
  </div>
</Sheet.Content>



<style>
  /* Ensure bullets and numbers are visible */
  #markdown-container :global(ul) {
    list-style-type: disc;
    margin-left: 1.5rem;
  }

  #markdown-container :global(ol) {
    list-style-type: decimal;
    margin-left: 1.5rem;
  }

  #markdown-container :global(li) {
    margin-bottom: 0.15rem;
    color: var(--muted-foreground);
  }

  /* Handle nested lists */
  #markdown-container :global(li > ul),
  #markdown-container :global(li > ol) {
    margin-top: 0.15rem;
    margin-left: 1.25rem;
  }
  
  #markdown-container :global(li > ul) {
    list-style-type: circle;
  }
  
  #markdown-container :global(li > ol) {
    list-style-type: lower-alpha;
  }
</style>