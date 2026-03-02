<script lang="ts">
  import '../app.css';
  import favicon from '$lib/assets/favicon.ico';

  let { children } = $props();
</script>

<svelte:head>
  <link rel="icon" href={favicon} />
  <title>Grading AI</title>
</svelte:head>

<div class="flex justify-center bg-fixed">
  <div class="big-container">
    {@render children()}
  </div>
</div>
