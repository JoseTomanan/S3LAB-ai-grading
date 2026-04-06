<script lang="ts">
  import logo from "$lib/assets/android-chrome-512x512.png";
  import * as Sheet from "$lib/components/ui/sheet/index.ts";
  import { Button, buttonVariants } from '$lib/components/CustomButton/index.ts';
  import { cn } from '$lib/utils.ts';

  import IconPaper from '~icons/mdi/paper';
  import IconSection from '~icons/mdi/person';
  import IconManual from '~icons/mdi/question-mark';
  import Manual from "./Manual.svelte";
</script>


<div class="container w-full h-[93dvh] flex flex-col items-center justify-center gap-y-10">
  <span class="pt-1"></span>
  
  <Sheet.Root>
    <Sheet.Trigger class={cn(buttonVariants({ variant: 'floating' }), 'absolute top-6 left-4')}>
      <IconManual class="size-8"/>
    </Sheet.Trigger>
    <Manual />
  </Sheet.Root>

  <div class="relative w-full py-8 flex flex-col items-center justify-center">
    <span class="absolute inset-0 m-auto size-28 object-contain -z-50 block left-0 right-0 top-0 bottom-0">
      <span class="slow-spin bg-accent/40 backdrop-blur-md rounded-full ring ring-primary-600/40">
        <enhanced:img src={logo}
              alt="Llama mascot"
              class="size-28"
          />
      </span>
    </span>
    <span class="text-white text-shadow-xs text-shadow-primary-700/40 text-5xl w-fit tracking-[-0.08em] leading-9 font-extrabold text-center">
      SIPAT<br>MATH
    </span>
  </div>
  
  <div class="space-y-3 w-full max-w-md px-4 sm:px-0">
    <div class="flex flex-col sm:flex-row gap-3 w-full
              [&>a]:flex-1 [&>a]:flex [&>a]:flex-col
              [&>a]:items-center [&>a]:gap-y-1
              [&>a]:py-4 [&>a]:px-4
              [&>a>svg]:size-10 [&>a>svg]:sm:size-12 [&>a>svg]:opacity-80
              [&>a>b]:font-bold [&>a>p]:text-xs [&>a>p]:text-muted-foreground
              ">
      <Button variant="outline"
              href="/instances">
        <IconPaper class="size-10"/>
        <b>Test Instances</b>
        <p>Manage exams & grading</p>
      </Button>
      <Button variant="outline"
              href="/sections">
        <IconSection class="size-10"/>
        <b>Sections</b>
        <p>Manage class list</p>
      </Button>
    </div>
  </div>
</div>

<h6 class="font-light opacity-50">Made for CS199 by Tomanan and Tuan, with love.</h6>
