<script lang="ts">
  import { page } from "$app/state";
</script>


<div class="px-4 py-8 flex flex-col gap-4">
  <h1>{page.status} {page.error?.message ?? ""}</h1>
  <h4>
    Please check the link you are visiting. <br>
    If you are attempting to visit a custom link, please start from the following: <br>
    <a href="/">Home</a> &middot;
    <a href="/sections">Sections</a>
  </h4>
</div>


<style>
  a {
    @apply underline;
  }
</style>